import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional
from models.seir_lstm import SEIR_LSTM


class SEIR_LSTM_Ensemble(nn.Module):
    """
    Ensemble of SEIR-LSTM models for improved prediction robustness.
    Combines multiple models with different architectures or training seeds.

    Supports:
    - Mean/Median/Weighted aggregation
    - Monte Carlo Dropout uncertainty quantification
    - Ensemble disagreement (epistemic uncertainty)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
        num_heads: int = 4,
        n_models: int = 3,
        aggregation_method: str = "mean",
    ):
        super(SEIR_LSTM_Ensemble, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.n_models = n_models
        self.aggregation_method = aggregation_method

        self.models = nn.ModuleList(
            [
                SEIR_LSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    num_heads=num_heads,
                )
                for _ in range(n_models)
            ]
        )

        if aggregation_method == "weighted_mean":
            self.aggregation_weights = nn.Parameter(torch.ones(n_models) / n_models)

    def forward(
        self,
        x: torch.Tensor,
        init_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        N: torch.Tensor,
        new_vacc_rate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")

        all_predictions = []
        all_betas = []
        all_sigmas = []
        all_gammas = []

        for model in self.models:
            preds, params = model(x, init_states, N, new_vacc_rate)
            all_predictions.append(preds)
            all_betas.append(params[0])
            all_sigmas.append(params[1])
            all_gammas.append(params[2])

        pred_stack = torch.stack(all_predictions)
        beta_stack = torch.stack(all_betas)
        sigma_stack = torch.stack(all_sigmas)
        gamma_stack = torch.stack(all_gammas)

        if self.aggregation_method == "median":
            pred_mean = torch.median(pred_stack, dim=0)[0]
            beta_mean = torch.median(beta_stack, dim=0)[0]
            sigma_mean = torch.median(sigma_stack, dim=0)[0]
            gamma_mean = torch.median(gamma_stack, dim=0)[0]
        elif self.aggregation_method == "weighted_mean":
            weights = F.softmax(self.aggregation_weights, dim=0)
            weights = weights.view(-1, 1, 1)
            pred_mean = torch.sum(weights * pred_stack, dim=0)
            beta_mean = torch.sum(weights * beta_stack, dim=0)
            sigma_mean = torch.sum(weights * sigma_stack, dim=0)
            gamma_mean = torch.sum(weights * gamma_stack, dim=0)
        else:  # mean
            pred_mean = torch.mean(pred_stack, dim=0)
            beta_mean = torch.mean(beta_stack, dim=0)
            sigma_mean = torch.mean(sigma_stack, dim=0)
            gamma_mean = torch.mean(gamma_stack, dim=0)

        return pred_mean, (beta_mean, sigma_mean, gamma_mean)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        init_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        N: torch.Tensor,
        new_vacc_rate: Optional[torch.Tensor] = None,
        n_samples: int = 100,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Make predictions with uncertainty quantification using Monte Carlo dropout
        and ensemble disagreement.

        Returns:
            Tuple of (pred_mean, pred_std, pred_std_epistemic, (beta, sigma, gamma))
            - pred_mean: Mean prediction across samples
            - pred_std: Total uncertainty (aleatoric + epistemic)
            - pred_std_epistemic: Model disagreement (epistemic only)
        """
        self.train()  # Enable dropout for MC sampling

        all_predictions = []
        all_betas = []
        all_sigmas = []
        all_gammas = []

        with torch.no_grad():
            for _ in range(n_samples):
                preds, params = self.forward(x, init_states, N, new_vacc_rate)
                all_predictions.append(preds)
                all_betas.append(params[0])
                all_sigmas.append(params[1])
                all_gammas.append(params[2])

        pred_stack = torch.stack(all_predictions)
        pred_mean = torch.mean(pred_stack, dim=0)
        pred_std = torch.std(pred_stack, dim=0)

        # Epistemic uncertainty: variance of ensemble means (model disagreement)
        # Split samples into groups, compute mean per group, then variance across groups
        n_groups = min(self.n_models, n_samples)
        group_size = n_samples // n_groups
        group_means = []
        for g in range(n_groups):
            group_pred = pred_stack[g * group_size : (g + 1) * group_size]
            group_means.append(torch.mean(group_pred, dim=0))
        pred_std_epistemic = torch.std(torch.stack(group_means), dim=0)

        beta_mean = torch.mean(torch.stack(all_betas), dim=0)
        sigma_mean = torch.mean(torch.stack(all_sigmas), dim=0)
        gamma_mean = torch.mean(torch.stack(all_gammas), dim=0)

        self.eval()
        return pred_mean, pred_std, pred_std_epistemic, (beta_mean, sigma_mean, gamma_mean)

    @classmethod
    def load_with_metadata(
        cls, path: str, device: torch.device
    ) -> Tuple["SEIR_LSTM_Ensemble", Dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        # Detect input_dim from state_dict if possible
        if "models.0.lstm.weight_ih_l0" in state_dict:
            input_dim = state_dict["models.0.lstm.weight_ih_l0"].shape[1]
        elif "models.0.input_proj.weight" in state_dict:
            input_dim = state_dict["models.0.input_proj.weight"].shape[1]
        else:
            input_dim = config.get("input_dim", 25)

        model = cls(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.25),
            num_heads=config.get("num_heads", 4),
            n_models=config.get("n_models", 3),
            aggregation_method=config.get("aggregation_method", "mean"),
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, config

    def save_with_metadata(self, path: str, config: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config["num_heads"] = self.num_heads
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": config,
            },
            path,
        )


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str], device: torch.device
) -> Tuple[SEIR_LSTM_Ensemble, Dict[str, Any]]:
    first_model, config = SEIR_LSTM.load_with_metadata(checkpoint_paths[0], device)

    ensemble = SEIR_LSTM_Ensemble(
        input_dim=config.get("input_dim", 17),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.25),
        num_heads=config.get("num_heads", 4),
        n_models=len(checkpoint_paths),
        aggregation_method="mean",
    )

    for i, path in enumerate(checkpoint_paths):
        model_state_dict = torch.load(path, map_location=device, weights_only=False)
        if "model_state_dict" in model_state_dict:
            model_state_dict = model_state_dict["model_state_dict"]
        ensemble.models[i].load_state_dict(model_state_dict)

    ensemble.to(device)
    ensemble.eval()

    return ensemble, config
