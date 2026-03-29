import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import logging
import json
import time
import warnings
from contextlib import nullcontext
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict

from config import PATHS, FEATURES, TRAIN_CONFIG, MODEL_CONFIG, get_device, set_seed
from models.seir_lstm import SEIR_LSTM

SCALER_PATH = PATHS.scaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainingHistory:
    def __init__(self, path: str):
        self.path = path
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_mape": [],
            "learning_rate": [],
            "epoch_time": [],
        }

    def update(
        self,
        train_loss: float,
        val_loss: float,
        val_mae: float,
        val_rmse: float,
        val_mape: float,
        lr: float,
        epoch_time: float,
    ):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_mae"].append(val_mae)
        self.history["val_rmse"].append(val_rmse)
        self.history["val_mape"].append(val_mape)
        self.history["learning_rate"].append(lr)
        self.history["epoch_time"].append(epoch_time)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)


class LogMAPELoss(nn.Module):
    """Combined loss: Log-space MSE + Huber loss for robustness to outliers.
    Log-space ensures equal relative error weighting across magnitudes."""

    def __init__(self, huber_delta: float = 50.0):
        super().__init__()
        self.huber_delta = huber_delta
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        # Log-space MSE for exponential growth dynamics
        log_pred = torch.log1p(pred.clamp(min=0))
        log_target = torch.log1p(target.clamp(min=0))
        loss_log = F.mse_loss(log_pred, log_target, reduction="none")

        # Huber loss in original space for robustness
        loss_huber = self.huber(pred.clamp(min=0), target)

        # Combined weighted loss
        combined = 0.7 * loss_log + 0.3 * loss_huber / (target.abs().mean() + 1e-6)
        return torch.mean(combined * weights.unsqueeze(0))


def compute_loss(
    model, X, y, S, E, I, R, N, V, weights, criterion, physics_weight, label_smoothing=0.0
):
    preds, params = model(X, (S, E, I, R), N, V)
    beta_t = params[0]
    sigma_t = params[1]
    gamma_t = params[2]

    # Ensure predictions are non-negative
    preds = torch.clamp(preds, min=0.0)

    # Label smoothing: blend target with its mean
    if label_smoothing > 0:
        y_smooth = y * (1 - label_smoothing) + label_smoothing * y.mean(dim=1, keepdim=True)
    else:
        y_smooth = y

    # Combined loss with temporal weighting
    loss_fit = criterion(preds, y_smooth, weights)

    # Physics-informed constraints
    device = preds.device
    loss_physics_total = torch.tensor(0.0, device=device)

    # 1. Temporal smoothness on beta (penalize abrupt parameter changes)
    if beta_t.size(1) > 1:
        loss_beta_smooth = torch.mean((beta_t[:, 1:] - beta_t[:, :-1]) ** 2)
        loss_sigma_smooth = torch.mean((sigma_t[:, 1:] - sigma_t[:, :-1]) ** 2)
        loss_gamma_smooth = torch.mean((gamma_t[:, 1:] - gamma_t[:, :-1]) ** 2)
        loss_physics_total = (
            loss_physics_total
            + (loss_beta_smooth + 0.5 * loss_sigma_smooth + 0.5 * loss_gamma_smooth)
            * physics_weight
        )

    # 2. Parameter bounds enforcement (soft penalty)
    beta_violation = torch.mean(F.relu(beta_t - 2.5)) + torch.mean(F.relu(0.0 - beta_t))
    sigma_violation = torch.mean(F.relu(sigma_t - 0.5)) + torch.mean(F.relu(0.05 - sigma_t))
    gamma_violation = torch.mean(F.relu(gamma_t - 0.3)) + torch.mean(F.relu(0.05 - gamma_t))
    loss_physics_total = (
        loss_physics_total
        + (beta_violation + sigma_violation + gamma_violation) * physics_weight * 0.5
    )

    # 3. Conservation constraint: compartments should sum to population
    if S.dim() > 1:
        compartment_sum = S + E + I + R
        total_pop = N if N.dim() == 1 else N[:, 0:1]
        conservation_error = torch.mean((compartment_sum - total_pop) ** 2)
        loss_physics_total = loss_physics_total + conservation_error * physics_weight * 0.2

    # 4. Monotonicity constraint on cumulative cases (new_cases should be non-negative)
    non_negative_penalty = torch.mean(F.relu(-preds))
    loss_physics_total = loss_physics_total + non_negative_penalty * physics_weight * 0.1

    return loss_fit + loss_physics_total, loss_fit.item(), preds


def prepare_data(
    df: pd.DataFrame,
    sequence_length: int = None,
    e_multiplier: float = None,
    save_scaler: bool = True,
) -> Tuple:
    features = FEATURES
    sequence_length = sequence_length or TRAIN_CONFIG.sequence_length
    e_multiplier = e_multiplier or TRAIN_CONFIG.e_multiplier

    train_df = df[df["date"] < TRAIN_CONFIG.train_split_date].copy()
    val_df = df[df["date"] >= TRAIN_CONFIG.train_split_date].copy()

    logger.info(
        f"Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | "
        f"Split date: {TRAIN_CONFIG.train_split_date}"
    )

    # Fit scaler only on training data to prevent data leakage
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])

    if save_scaler:
        os.makedirs(PATHS.saved_models, exist_ok=True)
        joblib.dump(scaler, PATHS.scaler)
        logger.info(f"Scaler saved to {PATHS.scaler}")

    def generate_windows(group_df, emult: float):
        X, y, S, E, I, R, N, V = [], [], [], [], [], [], [], []
        values = group_df[features].values
        cases = group_df["new_cases_smoothed"].values
        pop = group_df["population"].values[0] if not group_df.empty else 1e6
        total_c = group_df["total_cases"].values

        vacc_rate = group_df["new_vaccinations_smoothed"].values / max(pop, 1.0)

        group_len = len(group_df) - sequence_length
        if group_len <= 0:
            return X, y, S, E, I, R, N, V

        for i in range(group_len):
            I_0 = max(cases[i], 1.0)
            E_0 = I_0 * emult
            R_0 = max(total_c[i] - E_0 - I_0, 0)
            S_0 = max(pop - E_0 - I_0 - R_0, 0)

            X.append(values[i : i + sequence_length])
            y.append(cases[i : i + sequence_length])
            V.append(vacc_rate[i : i + sequence_length])
            S.append(S_0)
            E.append(E_0)
            I.append(I_0)
            R.append(R_0)
            N.append(pop)
        return X, y, S, E, I, R, N, V

    X_tr, y_tr, S_tr, E_tr, I_tr, R_tr, N_tr, V_tr = [], [], [], [], [], [], [], []
    X_v, y_v, S_v, E_v, I_v, R_v, N_v, V_v = [], [], [], [], [], [], [], []

    for _, group in train_df.groupby("Country/Region"):
        res = generate_windows(group, e_multiplier)
        X_tr.extend(res[0])
        y_tr.extend(res[1])
        S_tr.extend(res[2])
        E_tr.extend(res[3])
        I_tr.extend(res[4])
        R_tr.extend(res[5])
        N_tr.extend(res[6])
        V_tr.extend(res[7])

    for _, group in val_df.groupby("Country/Region"):
        res = generate_windows(group, e_multiplier)
        X_v.extend(res[0])
        y_v.extend(res[1])
        S_v.extend(res[2])
        E_v.extend(res[3])
        I_v.extend(res[4])
        R_v.extend(res[5])
        N_v.extend(res[6])
        V_v.extend(res[7])

    logger.info(f"Windows generated: train={len(X_tr):,} | val={len(X_v):,}")

    def to_tensor(arr: List) -> torch.Tensor:
        return torch.tensor(np.array(arr), dtype=torch.float32)

    return tuple(
        to_tensor(u)
        for u in [
            X_tr,
            y_tr,
            S_tr,
            E_tr,
            I_tr,
            R_tr,
            N_tr,
            V_tr,
            X_v,
            y_v,
            S_v,
            E_v,
            I_v,
            R_v,
            N_v,
            V_v,
        ]
    )


def train(epochs: Optional[int] = None, seed: int = 42) -> None:
    set_seed(seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    epochs = epochs or TRAIN_CONFIG.epochs

    if not os.path.exists(PATHS.merged_data):
        logger.error(f"Training data not found at {PATHS.merged_data}. Run preprocessing first.")
        raise FileNotFoundError(f"Missing: {PATHS.merged_data}")

    logger.info("Loading data for training...")
    df = pd.read_csv(PATHS.merged_data)

    if df["population"].isna().any():
        missing_pop = df[df["population"].isna()]["Country/Region"].unique()
        logger.warning(f"Missing population for countries: {missing_pop}")
        df = df.dropna(subset=["population"])

    if df.empty:
        raise ValueError("No data remaining after dropping missing population values")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["Country/Region", "date"])

    min_date = df["date"].min()
    max_date = df["date"].max()
    train_split = pd.to_datetime(TRAIN_CONFIG.train_split_date)
    if train_split < min_date or train_split > max_date:
        logger.warning(
            f"Train split date {train_split} outside data range [{min_date}, {max_date}]"
        )

    logger.info("Preparing tensors (Time-Series Split by Date)...")
    (
        X_tr,
        y_tr,
        S_tr,
        E_tr,
        I_tr,
        R_tr,
        N_tr,
        V_tr,
        X_v,
        y_v,
        S_v,
        E_v,
        I_v,
        R_v,
        N_v,
        V_v,
    ) = prepare_data(
        df,
        e_multiplier=TRAIN_CONFIG.e_multiplier,
        save_scaler=True,
    )

    # Dynamically set input_dim based on actual features in the data
    actual_input_dim = X_tr.shape[2]
    if MODEL_CONFIG.input_dim == 0 or MODEL_CONFIG.input_dim != actual_input_dim:
        MODEL_CONFIG.input_dim = actual_input_dim
        logger.info(f"Set model input_dim to {actual_input_dim} based on actual features")

    train_data = TensorDataset(X_tr, y_tr, S_tr, E_tr, I_tr, R_tr, N_tr, V_tr)
    train_loader = DataLoader(
        train_data,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
    )

    val_data = TensorDataset(X_v, y_v, S_v, E_v, I_v, R_v, N_v, V_v)
    val_loader = DataLoader(
        val_data, batch_size=TRAIN_CONFIG.batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True
    )

    if len(train_loader) == 0:
        raise ValueError(
            "No training windows produced. Check merged_data, sequence_length, and filters."
        )
    if len(val_loader) == 0:
        raise ValueError(
            "No validation windows after the train split. "
            "Try an earlier train_split_date or a shorter sequence_length."
        )

    model = SEIR_LSTM(
        input_dim=MODEL_CONFIG.input_dim,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        num_layers=MODEL_CONFIG.num_layers,
        dropout=MODEL_CONFIG.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total parameters, {trainable_params:,} trainable")

    # Use combined log-Huber loss for robustness
    criterion = LogMAPELoss(huber_delta=50.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG.learning_rate,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )

    warmup_epochs = TRAIN_CONFIG.warmup_epochs
    total_steps = epochs
    warmup_steps = warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision with modern API
    use_amp = device.type == "cuda"
    amp_context = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    accumulation_steps = TRAIN_CONFIG.gradient_accumulation_steps
    effective_batch_size = TRAIN_CONFIG.batch_size * accumulation_steps

    # Exponential weights favoring recent timesteps in the sequence
    weights = torch.exp(torch.arange(TRAIN_CONFIG.sequence_length, dtype=torch.float32) / 10.0)
    weights = weights / weights.sum() * float(TRAIN_CONFIG.sequence_length)
    weights = weights.to(device)

    logger.info(f"Starting Training... (Effective batch size: {effective_batch_size})")
    best_val_loss = float("inf")
    patience, patience_counter = TRAIN_CONFIG.patience, 0

    history = TrainingHistory(os.path.join(PATHS.saved_models, "training_history.json"))

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            X_b, y_b, S_b, E_b, I_b, R_b, N_b, V_b = (b.to(device) for b in batch)

            with amp_context:
                loss, _, _ = compute_loss(
                    model,
                    X_b,
                    y_b,
                    S_b,
                    E_b,
                    I_b,
                    R_b,
                    N_b,
                    V_b,
                    weights,
                    criterion,
                    TRAIN_CONFIG.physics_loss_weight,
                    TRAIN_CONFIG.label_smoothing,
                )
                loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=TRAIN_CONFIG.max_grad_norm
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        # Handle remaining gradients from incomplete accumulation cycle
        remaining_steps = len(train_loader) % accumulation_steps
        if remaining_steps > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAIN_CONFIG.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse_sum = 0
        val_mape_sum = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                X_b, y_b, S_b, E_b, I_b, R_b, N_b, V_b = (b.to(device) for b in batch)

                preds, _ = model(X_b, (S_b, E_b, I_b, R_b), N_b, V_b)
                preds_clamped = torch.clamp(preds, min=0.0)
                loss = criterion(preds_clamped, y_b, weights)
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(preds_clamped - y_b)).item()
                val_mse_sum += torch.mean((preds_clamped - y_b) ** 2).item()
                # MAPE: clip small values to avoid extreme percentages
                # Use max(y, 1) as minimum to handle near-zero cases in epidemic data
                mape_denom = torch.clamp(y_b.abs(), min=1.0)
                val_mape_sum += torch.mean(torch.abs(preds_clamped - y_b) / mape_denom).item() * 100
                val_count += 1

        train_loss /= len(train_loader)
        val_loss /= max(val_count, 1)
        val_mae /= max(val_count, 1)
        val_rmse = np.sqrt(val_mse_sum / max(val_count, 1)) if val_count > 0 else 0.0
        val_mape = val_mape_sum / max(val_count, 1)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.1f} | "
            f"Val RMSE: {val_rmse:.1f} | "
            f"(MAPE: {val_mape:.1f}% - use MAE/RMSE for decisions) | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history.update(train_loss, val_loss, val_mae, val_rmse, val_mape, current_lr, epoch_time)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_with_metadata(PATHS.model_weights, MODEL_CONFIG.to_dict())
            patience_counter = 0
            logger.info(f"  -> New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                checkpoint_path = PATHS.model_weights.replace(".pth", f"_epoch{epoch + 1}.pth")
                model.save_with_metadata(checkpoint_path, MODEL_CONFIG.to_dict())
                logger.info(f"  -> Periodic checkpoint saved at epoch {epoch + 1}")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    history.save()
    logger.info(f"Training finished. Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Training history saved to {PATHS.saved_models}/training_history.json")


if __name__ == "__main__":
    train()
