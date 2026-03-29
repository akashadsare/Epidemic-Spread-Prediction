import os
import torch
import logging

from config import PATHS, TRAIN_CONFIG, MODEL_CONFIG, get_device
from train import train
from models.ensemble import SEIR_LSTM_Ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_ensemble(n_models: int = 3):
    device = get_device()
    logger.info(f"Training ensemble of {n_models} models on {device}...")

    if not os.path.exists(PATHS.merged_data):
        logger.error("Merged data not found. Run preprocessing first.")
        return

    checkpoint_paths = []

    for i in range(n_models):
        logger.info(f"--- Training model {i + 1}/{n_models} ---")
        model_path = PATHS.model_weights.replace(".pth", f"_member_{i}.pth")

        # We call the train function but we need it to save to a specific path
        # Let's modify the train function to accept a save path,
        # or just do the training loop here.
        # To avoid code duplication, let's just use different seeds.

        seed = 42 + i
        # We need a way to tell the 'train' function where to save.
        # For now, let's just manually run a simplified version or
        # temporarily monkeypatch PATHS.model_weights.

        original_path = PATHS.model_weights
        PATHS.model_weights = model_path

        try:
            train(epochs=TRAIN_CONFIG.epochs, seed=seed)
            checkpoint_paths.append(model_path)
        finally:
            PATHS.model_weights = original_path

    logger.info("Creating ensemble from trained checkpoints...")
    ensemble = SEIR_LSTM_Ensemble(
        input_dim=MODEL_CONFIG.input_dim,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        num_layers=MODEL_CONFIG.num_layers,
        dropout=MODEL_CONFIG.dropout,
        num_heads=4,
        n_models=n_models,
        aggregation_method="mean",
    )

    for i, path in enumerate(checkpoint_paths):
        checkpoint = torch.load(path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        )
        ensemble.models[i].load_state_dict(state_dict)

    ensemble_path = PATHS.model_weights.replace(".pth", "_ensemble.pth")
    ensemble.save_with_metadata(ensemble_path, MODEL_CONFIG.to_dict())
    logger.info(f"Ensemble saved to {ensemble_path}")


if __name__ == "__main__":
    train_ensemble(n_models=3)
