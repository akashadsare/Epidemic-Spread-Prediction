import os
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Core features always used
CORE_FEATURES: List[str] = [
    "retail_and_recreation_percent_change_from_baseline_smoothed",
    "grocery_and_pharmacy_percent_change_from_baseline_smoothed",
    "parks_percent_change_from_baseline_smoothed",
    "transit_stations_percent_change_from_baseline_smoothed",
    "workplaces_percent_change_from_baseline_smoothed",
    "residential_percent_change_from_baseline_smoothed",
    "people_fully_vaccinated",
    "new_tests_smoothed",
    "human_development_index",
    "hospital_beds_per_thousand",
    "stringency_index",
]

# Log-difference features for epidemic dynamics (GPR paper recommendation)
LOG_DIFF_FEATURES: List[str] = [
    "log_diff_7",
    "log_diff_14",
    "log_diff_30",
    "log_diff_7_30d",
    "log_diff_14_30d",
    "growth_rate_logdiff_7",
    "growth_rate_logdiff_14",
    "log_cases",
    "log_total_cases",
    "trend_acceleration",
    "velocity",
    "relative_change_7d",
    "relative_change_14d",
]

# Environmental features (synthetic data for seasonal analysis)
ENVIRONMENTAL_FEATURES: List[str] = [
    "temperature_avg",
    "temperature_min",
    "temperature_max",
    "humidity_avg",
    "precipitation",
    "uv_index",
]

FEATURES: List[str] = CORE_FEATURES + LOG_DIFF_FEATURES + ENVIRONMENTAL_FEATURES


@dataclass
class PathConfig:
    jhu_cases: str = ""
    owid: str = ""
    google_mobility: str = ""
    processed_data: str = ""
    saved_models: str = ""
    merged_data: str = ""
    model_weights: str = ""
    scaler: str = ""
    country_matches: str = ""
    evaluation_plots: str = ""
    evaluation_results: str = ""
    training_history: str = ""

    def __post_init__(self):
        if not self.jhu_cases:
            self.jhu_cases = os.path.join(
                BASE_DIR,
                "COVID-19",
                "csse_covid_19_data",
                "csse_covid_19_time_series",
                "time_series_covid19_confirmed_global.csv",
            )
        if not self.owid:
            self.owid = os.path.join(BASE_DIR, "covid-19-data", "public", "data", "owid-covid-data.csv")
        if not self.google_mobility:
            self.google_mobility = os.path.join(BASE_DIR, "Global_Mobility_Report.csv")
        if not self.processed_data:
            self.processed_data = os.path.join(BASE_DIR, "processed_data")
        if not self.saved_models:
            self.saved_models = os.path.join(BASE_DIR, "saved_models")
        if not self.merged_data:
            self.merged_data = os.path.join(self.processed_data, "merged_data.csv")
        if not self.model_weights:
            self.model_weights = os.path.join(self.saved_models, "hybrid_model.pth")
        if not self.scaler:
            self.scaler = os.path.join(self.saved_models, "scaler.pkl")
        if not self.country_matches:
            self.country_matches = os.path.join(self.processed_data, "country_matches.json")
        if not self.evaluation_plots:
            self.evaluation_plots = os.path.join(BASE_DIR, "evaluation_plots")
        if not self.evaluation_results:
            self.evaluation_results = os.path.join(BASE_DIR, "evaluation_results.csv")
        if not self.training_history:
            self.training_history = os.path.join(self.saved_models, "training_history.json")


@dataclass
class TrainConfig:
    sequence_length: int = 30
    forecast_horizon: int = 30
    train_split_date: str = "2022-10-01"
    batch_size: int = 8192
    epochs: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 7
    min_cases_threshold: int = 5000
    e_multiplier: float = 3.0
    physics_loss_weight: float = 0.15
    warmup_epochs: int = 3
    gradient_accumulation_steps: int = 2
    label_smoothing: float = 0.05
    ema_decay: float = 0.999
    max_grad_norm: float = 1.0
    scheduler_type: str = "cosine_warmup"
    smoothing_window: int = 30
    log_diff_eta: int = 7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    input_dim: int = 0  # Will be set dynamically based on actual data
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.25
    attention_heads: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SEIRParams:
    beta_max: float = 2.0
    sigma_range: tuple = (0.1, 0.4)
    gamma_range: tuple = (0.05, 0.25)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PATHS = PathConfig()
TRAIN_CONFIG = TrainConfig()
MODEL_CONFIG = ModelConfig()
SEIR_PARAMS = SEIRParams()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
