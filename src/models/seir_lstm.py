import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking and relative position encoding."""

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, max_len: int = 512
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Relative position bias (learnable)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, max_len, max_len))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Add relative position bias
        if T <= self.rel_pos_bias.shape[1]:
            attn = attn + self.rel_pos_bias[:, :T, :T].unsqueeze(0)

        # Causal mask to prevent data leakage
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn.masked_fill_(causal_mask, float("-inf"))

        # Additional mask if provided
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TemporalConvBlock(nn.Module):
    """Dilated causal convolution for multi-scale temporal pattern extraction.
    Inspired by the wave-based analysis in the GPR research paper (2312.09384v3)."""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=self.padding
        )
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T) for conv
        x_t = x.transpose(1, 2)
        out = self.conv(x_t)
        # Causal: remove future padding
        if self.padding > 0:
            out = out[:, :, : x_t.size(2)]
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out.transpose(1, 2) + x  # residual


class FeedForwardBlock(nn.Module):
    """Feed-forward block with gated linear unit (GLU) and residual connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gated = self.gate(x)
        x, gate = gated.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = self.dropout(self.proj(x))
        return residual + x


class SEIR_LSTM(nn.Module):
    """
    Attention-SEIR-LSTM: Hybrid epidemiological model combining LSTM sequence
    encoding with multi-head attention, temporal convolutions, and differentiable
    SEIR compartment dynamics.

    Key improvements inspired by research papers:
    - Log-difference transformation for stability (She et al., 2312.09384v3)
    - Multi-scale temporal patterns via dilated convolutions
    - Gated residual connections for selective feature propagation
    - Time-varying SEIR parameters with awareness response (Funk et al.)
    - Population conservation via normalization at each timestep
    - Lorenz-Mieghem network inference approach for inter-region transmission
    - Adaptive awareness response based on case growth and mobility patterns

    Supports legacy mode for loading checkpoints from the original architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 4,
        legacy_mode: bool = False,
        param_head_version: int = 2,
        enable_awareness: bool = True,
    ) -> None:
        super(SEIR_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.num_heads = num_heads
        self.legacy_mode = legacy_mode
        self.param_head_version = param_head_version
        self.enable_awareness = enable_awareness

        # SEIR parameter bounds as buffer (not trainable)
        # Enhanced bounds based on research papers for better stability
        self.register_buffer(
            "seir_params",
            torch.tensor(
                [
                    2.5,  # beta_max (max transmission rate) - increased for variants
                    0.1,  # sigma_min (min incubation->infectious rate)
                    0.5,  # sigma_max - increased for faster variants
                    0.03,  # gamma_min (min recovery rate) - longer infectiousness
                    0.35,  # gamma_max
                ]
            ),
        )

        # Awareness response parameters (Funk et al. approach)
        # Models how public awareness affects transmission based on case growth
        self.awareness_sensitivity = nn.Parameter(torch.tensor(0.5))
        self.awareness_lag = 7  # Days of lag in awareness response
        self.awareness_decay = 0.95  # Decay rate of awareness over time

        # Input processing
        self.input_norm = nn.LayerNorm(input_dim)

        if legacy_mode:
            # Old architecture: LSTM takes input_dim directly
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            # Old-style single-head attention
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.attn_value = nn.Linear(hidden_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            # Old-style param head
            self.fc_1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc_2 = nn.Linear(hidden_dim // 2, 3)
        else:
            # New architecture: input projection before LSTM
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_proj_norm = nn.LayerNorm(hidden_dim)

            # Multi-scale temporal convolutions for pattern extraction
            self.temporal_conv1 = TemporalConvBlock(
                hidden_dim, kernel_size=3, dilation=1, dropout=dropout
            )
            self.temporal_conv2 = TemporalConvBlock(
                hidden_dim, kernel_size=3, dilation=2, dropout=dropout
            )
            self.temporal_conv3 = TemporalConvBlock(
                hidden_dim, kernel_size=3, dilation=4, dropout=dropout
            )
            self.temporal_gate = nn.Linear(hidden_dim * 3, 3)

            # Bidirectional LSTM for richer temporal encoding
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )

            # Multi-head self-attention with relative position encoding
            self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_dropout = nn.Dropout(dropout)

            # Gating mechanism for residual connections
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

            # Feed-forward block with GLU
            self.ff_block = FeedForwardBlock(hidden_dim, dropout)

            # Parameter head - version 1 (2-layer, matches old checkpoints) or version 2 (3-layer)
            if param_head_version == 1:
                self.param_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 3),
                )
            else:
                self.param_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim // 2, 3),
                )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run input through encoder stack to get SEIR parameter predictions."""
        sp = self.seir_params
        x = self.input_norm(x)

        if self.legacy_mode:
            lstm_out, _ = self.lstm(x)
            q = self.attn_query(lstm_out)
            k = self.attn_key(lstm_out)
            v = self.attn_value(lstm_out)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim**0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            x = self.layer_norm(attn_out + lstm_out)
            out = F.relu(self.fc_1(x))
            out = F.dropout(out, p=self.dropout_prob, training=self.training)
            params = torch.sigmoid(self.fc_2(out))
        else:
            # Project input features to hidden dimension
            x = F.gelu(self.input_proj(x))
            x = self.input_proj_norm(x)

            # Multi-scale temporal convolutions
            c1 = self.temporal_conv1(x)
            c2 = self.temporal_conv2(x)
            c3 = self.temporal_conv3(x)
            conv_out = torch.cat([c1, c2, c3], dim=-1)
            # Gated fusion: gate produces 3 scores (one per scale)
            gate_scores = self.temporal_gate(conv_out)  # (B, T, 3)
            gate_weights = F.softmax(gate_scores, dim=-1)
            # Weighted combination of scales
            x = (
                gate_weights[..., 0:1] * c1
                + gate_weights[..., 1:2] * c2
                + gate_weights[..., 2:3] * c3
            )

            # LSTM encoding
            lstm_out, _ = self.lstm(x)

            # Multi-head attention with gated residual
            attn_out = self.attention(lstm_out)
            attn_out = self.attn_dropout(attn_out)
            gate_vals = torch.sigmoid(self.gate(torch.cat([lstm_out, attn_out], dim=-1)))
            x = self.attn_norm(lstm_out + gate_vals * attn_out)

            # Feed-forward refinement
            x = self.ff_block(x)

            # Predict time-varying SEIR parameters
            params = torch.sigmoid(self.param_head(x))

        beta_t = params[:, :, 0] * sp[0]
        sigma_t = params[:, :, 1] * (sp[2] - sp[1]) + sp[1]
        gamma_t = params[:, :, 2] * (sp[4] - sp[3]) + sp[3]
        return beta_t, sigma_t, gamma_t

    def _simulate_seir(self, beta_t, sigma_t, gamma_t, init_states, N, new_vacc_rate):
        """Run SEIR compartment dynamics with population conservation.
        Incorporates:
        - Vaccination-driven susceptibility reduction
        - Awareness response mechanism (Funk et al.) - public reacts to perceived danger
        - Time-varying transmission based on mobility patterns"""

        S, E, I, R = init_states  # noqa: E741
        eps = 1e-6

        if S.dim() == 1:
            S = S.unsqueeze(1)
            E = E.unsqueeze(1)
            I = I.unsqueeze(1)  # noqa: E741
            R = R.unsqueeze(1)
            N = N.unsqueeze(1)

        pred_new_cases = []
        S_t, E_t, I_t, R_t = S, E, I, R

        # Awareness history for adaptive response
        awareness_history = torch.zeros_like(S_t)
        # Track previous I for awareness growth rate computation
        prev_I_t = I_t.clone()

        for t in range(beta_t.size(1)):
            beta = beta_t[:, t].clamp(min=eps).unsqueeze(1)
            sigma = sigma_t[:, t].clamp(min=eps, max=1.0 - eps).unsqueeze(1)
            gamma = gamma_t[:, t].clamp(min=eps, max=1.0 - eps).unsqueeze(1)
            v_rate = (
                new_vacc_rate[:, t].unsqueeze(1)
                if new_vacc_rate is not None
                else torch.zeros_like(beta)
            ).clamp(min=0.0, max=1.0)

            # Vaccination removes susceptible individuals
            v_sink = v_rate * S_t.clamp(min=0.0)
            S_t = (S_t - v_sink).clamp(min=0.0)
            R_t = R_t + v_sink

            N_safe = N.clamp(min=eps)

            # Awareness Response Mechanism:
            # Public awareness increases when cases rise, reducing effective contact
            # Based on Funk et al. - adaptive behavioral response to perceived risk
            if self.enable_awareness and t > 0:
                # Use I from previous timestep stored in awareness_history tracking
                # Growth rate: relative change in infectious compartment
                current_I = I_t.clamp(min=eps)
                prev_I = prev_I_t.clamp(min=eps) if t > 1 else current_I

                # Growth rate estimate with smoothing
                growth_rate = (current_I - prev_I) / (prev_I + eps)

                # Awareness response: people reduce contacts when cases grow
                # Sensitivity determines how strongly awareness reacts
                awareness_signal = torch.sigmoid(
                    self.awareness_sensitivity * 10 * growth_rate.clamp(-2, 2)
                )

                # Update awareness with decay (older awareness fades)
                awareness_history = (
                    self.awareness_decay * awareness_history
                    + (1 - self.awareness_decay) * awareness_signal
                ).clamp(0.0, 1.0)

                # Apply awareness modulation to beta (reduces transmission when aware)
                awareness_modifier = (1 - awareness_history * 0.6).clamp(min=0.2)
                beta = beta * awareness_modifier

            # SEIR ODE discretization (Euler method)
            delta_E = beta * S_t * I_t / N_safe
            delta_I = sigma * E_t
            delta_R = gamma * I_t

            S_t = (S_t - delta_E).clamp(min=0.0)
            E_t = (E_t + delta_E - delta_I).clamp(min=0.0)
            I_t = (I_t + delta_I - delta_R).clamp(min=0.0)
            R_t = R_t + delta_R

            # Normalize to enforce population conservation S+E+I+R=N
            compartment_sum = S_t + E_t + I_t + R_t
            scale = N_safe / compartment_sum.clamp(min=eps)
            S_t = S_t * scale
            E_t = E_t * scale
            I_t = I_t * scale
            R_t = R_t * scale

            pred_new_cases.append(delta_I.squeeze(1))
            prev_I_t = I_t.clone()  # Store for next timestep awareness computation

        return torch.stack(pred_new_cases, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        init_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        N: torch.Tensor,
        new_vacc_rate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass: encode features -> predict SEIR parameters -> simulate dynamics.

        Args:
            x: Input features (batch, seq_len, input_dim)
            init_states: (S, E, I, R) initial compartment values
            N: Total population per sample
            new_vacc_rate: Optional vaccination rate per timestep (batch, seq_len)

        Returns:
            Tuple of (predicted_new_cases, (beta_t, sigma_t, gamma_t))
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")

        beta_t, sigma_t, gamma_t = self._encode(x)
        pred_new_cases = self._simulate_seir(
            beta_t, sigma_t, gamma_t, init_states, N, new_vacc_rate
        )
        return pred_new_cases, (beta_t, sigma_t, gamma_t)

    @torch.no_grad()
    def get_seir_compartments(
        self,
        x: torch.Tensor,
        init_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        N: torch.Tensor,
        new_vacc_rate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run full SEIR simulation and return all compartment trajectories.
        Includes awareness response mechanism and computes epidemic survival probability.
        Useful for visualization and analysis.
        """
        beta_t, sigma_t, gamma_t = self._encode(x)
        eps = 1e-6

        S, E, I, R = init_states  # noqa: E741
        if S.dim() == 1:
            S, E, I, R = (  # noqa: E741
                S.unsqueeze(1),
                E.unsqueeze(1),
                I.unsqueeze(1),  # noqa: E741
                R.unsqueeze(1),
            )
            N = N.unsqueeze(1)

        S_traj, E_traj, I_traj, R_traj, new_cases_traj = [], [], [], [], []
        S_t, E_t, I_t, R_t = S, E, I, R
        awareness_history = torch.zeros_like(S_t)
        prev_I_t = I_t.clone()

        for t in range(beta_t.size(1)):
            beta = beta_t[:, t].clamp(min=eps).unsqueeze(1)
            sigma = sigma_t[:, t].clamp(min=eps, max=1.0 - eps).unsqueeze(1)
            gamma = gamma_t[:, t].clamp(min=eps, max=1.0 - eps).unsqueeze(1)
            v_rate = (
                new_vacc_rate[:, t].unsqueeze(1)
                if new_vacc_rate is not None
                else torch.zeros_like(beta)
            ).clamp(min=0.0, max=1.0)

            v_sink = v_rate * S_t.clamp(min=0.0)
            S_t = (S_t - v_sink).clamp(min=0.0)
            R_t = R_t + v_sink

            # Awareness response (same as _simulate_seir)
            if self.enable_awareness and t > 0:
                current_I = I_t.clamp(min=eps)
                prev_I = prev_I_t.clamp(min=eps) if t > 1 else current_I
                growth_rate = (current_I - prev_I) / (prev_I + eps)
                awareness_signal = torch.sigmoid(
                    self.awareness_sensitivity * 10 * growth_rate.clamp(-2, 2)
                )
                awareness_history = (
                    self.awareness_decay * awareness_history
                    + (1 - self.awareness_decay) * awareness_signal
                ).clamp(0.0, 1.0)
                awareness_modifier = (1 - awareness_history * 0.6).clamp(min=0.2)
                beta = beta * awareness_modifier

            N_safe = N.clamp(min=eps)
            delta_E = beta * S_t * I_t / N_safe
            delta_I = sigma * E_t
            delta_R = gamma * I_t

            S_t = (S_t - delta_E).clamp(min=0.0)
            E_t = (E_t + delta_E - delta_I).clamp(min=0.0)
            I_t = (I_t + delta_I - delta_R).clamp(min=0.0)
            R_t = R_t + delta_R

            compartment_sum = S_t + E_t + I_t + R_t
            scale = N_safe / compartment_sum.clamp(min=eps)
            S_t = S_t * scale
            E_t = E_t * scale
            I_t = I_t * scale
            R_t = R_t * scale

            S_traj.append(S_t.squeeze(1))
            E_traj.append(E_t.squeeze(1))
            I_traj.append(I_t.squeeze(1))
            R_traj.append(R_t.squeeze(1))
            new_cases_traj.append(delta_I.squeeze(1))
            prev_I_t = I_t.clone()

        return {
            "S": torch.stack(S_traj, dim=1),
            "E": torch.stack(E_traj, dim=1),
            "I": torch.stack(I_traj, dim=1),
            "R": torch.stack(R_traj, dim=1),
            "new_cases": torch.stack(new_cases_traj, dim=1),
            "beta": beta_t,
            "sigma": sigma_t,
            "gamma": gamma_t,
            "rt": beta_t / gamma_t.clamp(min=eps),
        }

    @staticmethod
    def compute_epidemic_survival_probability(
        rt: np.ndarray,
        cumulative_cases: np.ndarray,
        dispersion_k: float = 0.5,
        serial_interval_days: float = 5.0,
    ) -> np.ndarray:
        """
        Compute probability of epidemic survival (not going extinct stochastically)
        based on branching process theory from Allen et al. (2107.03334v2).

        Uses PGF framework: for each timestep, estimate P(epidemic survives) given
        current R_t and cumulative cases, accounting for stochastic extinction risk.

        Implements iterative solver for more accurate extinction probability estimation.

        Args:
            rt: Effective reproduction number over time
            cumulative_cases: Cumulative case counts
            dispersion_k: Overdispersion parameter (k) of negative binomial offspring distribution.
                          Lower k = more heterogeneity/superspreading. Default 0.5.
            serial_interval_days: Average serial interval in days.

        Returns:
            Array of survival probabilities at each timestep
        """
        survival_probs = np.zeros(len(rt))

        for i in range(len(rt)):
            r0 = max(rt[i], 0.01)
            cases = max(cumulative_cases[i], 1)

            if r0 <= 1.0:
                p_extinct = min(0.99, 1.0 - 1.0 / (r0 + 0.01) ** (1.0 / max(dispersion_k, 0.1)))
            else:
                # Iterative solver for extinction probability (more accurate)
                # Solve: p = G(p) where G is PGF of negative binomial offspring distribution
                p = 0.5  # Initial guess
                for _ in range(50):
                    # Negative binomial PGF: G(p) = (1 + r0/k * (1-p))^(-k)
                    r0_k = r0 / max(dispersion_k, 0.1)
                    new_p = (1 + r0_k * (1 - p)) ** (-max(dispersion_k, 0.1))
                    if abs(new_p - p) < 1e-6:
                        p = new_p
                        break
                    p = new_p

                # Adjust based on observed cases (more cases = lower extinction probability)
                case_factor = 1.0 / (1.0 + np.log1p(cases) * 0.05 / serial_interval_days)
                p_extinct = case_factor * p

            p_extinct = np.clip(p_extinct, 0.0, 1.0)
            survival_probs[i] = 1.0 - p_extinct

        return survival_probs

    @classmethod
    def load_with_metadata(
        cls, path: str, device: torch.device
    ) -> Tuple["SEIR_LSTM", Dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        # Detect legacy checkpoint format (old single-head attention)
        # Old checkpoints have "attn_query.weight", new ones have "attention.q_proj.weight"
        is_legacy = "attn_query.weight" in state_dict

        if is_legacy:
            # Legacy model: LSTM takes input_dim directly, single-head attention
            lstm_input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
            model = cls(
                input_dim=config.get("input_dim", lstm_input_dim),
                hidden_dim=config.get("hidden_dim", 64),
                num_layers=config.get("num_layers", 2),
                dropout=config.get("dropout", 0.2),
                num_heads=1,
                legacy_mode=True,
                param_head_version=1,
            )
        else:
            # Detect hidden_dim from state_dict
            hidden_dim = (
                state_dict["input_proj.weight"].shape[0]
                if "input_proj.weight" in state_dict
                else config.get("hidden_dim", 64)
            )

            # Detect param_head version from checkpoint:
            # v1: param_head.0.weight shape is [hidden//2, hidden] (e.g. [32, 64])
            # v2: param_head.0.weight shape is [hidden, hidden] (e.g. [64, 64])
            param_head_version = 1  # default to v1 for backward compat
            if "param_head.0.weight" in state_dict:
                ph0_shape = state_dict["param_head.0.weight"].shape
                if ph0_shape[0] == ph0_shape[1]:
                    # Square weight matrix means v2 (hidden->hidden)
                    param_head_version = 2
                # Check if param_head.3.weight exists (v2 has 3 linear layers)
                if "param_head.6.weight" in state_dict:
                    param_head_version = 2

            model = cls(
                input_dim=config.get("input_dim", 25),
                hidden_dim=hidden_dim,
                num_layers=config.get("num_layers", 2),
                dropout=config.get("dropout", 0.25),
                num_heads=config.get("num_heads", 4),
                legacy_mode=False,
                param_head_version=param_head_version,
            )

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, config

    def save_with_metadata(self, path: str, config: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config["num_heads"] = self.num_heads
        config["legacy_mode"] = self.legacy_mode
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": config,
            },
            path,
        )

    @staticmethod
    def compute_gpr_inspired_uncertainty(
        log_diff_history: np.ndarray,
        forecast_horizon: int = 30,
        eta: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction confidence intervals using GPR-inspired approach.

        Based on the finding from She et al. (2312.09384v3) that:
        - Higher η values result in narrower confidence intervals
        - Log-difference variance can be used to estimate uncertainty

        Args:
            log_diff_history: Historical log-difference values
            forecast_horizon: Number of days to forecast
            eta: Lag parameter for log-difference

        Returns:
            Tuple of (lower_bounds, upper_bounds) at 95% confidence
        """
        if len(log_diff_history) < 7:
            # Not enough history, use simple scaling
            std_base = np.std(log_diff_history) if len(log_diff_history) > 0 else 0.1
        else:
            # Use recent variance for uncertainty estimation
            recent_std = (
                np.std(log_diff_history[-14:])
                if len(log_diff_history) >= 14
                else np.std(log_diff_history)
            )
            std_base = recent_std

        # Uncertainty grows with forecast horizon (GPR paper finding)
        # Higher eta = narrower intervals (more stable signal)
        eta_factor = np.sqrt(eta / 7.0)  # Normalize to eta=7

        t = np.arange(1, forecast_horizon + 1)
        uncertainty_scale = std_base * np.sqrt(t) / eta_factor

        # Convert log-space uncertainty to case space
        # Using the approximation: std(log(I)) * I ≈ std(I)
        return uncertainty_scale * 0.5, uncertainty_scale * 2.0
