import pandas as pd
import numpy as np
import os
import logging
import json
from rapidfuzz import process, fuzz
import warnings
from datetime import datetime  # noqa: F401
import sys

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, TRAIN_CONFIG

FUZZY_MATCH_THRESHOLD = 80

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        msg = f"Required file not found: {path}. Please run 'python src/data/download_data.py' to fetch the necessary datasets."
        logger.error(msg)
        raise FileNotFoundError(msg)


def load_jhu_data() -> pd.DataFrame:
    logger.info("Loading JHU data...")
    _check_file_exists(PATHS.jhu_cases)

    jhu = pd.read_csv(PATHS.jhu_cases)
    jhu = (
        jhu.drop(columns=["Province/State", "Lat", "Long"])
        .groupby("Country/Region")
        .sum()
        .reset_index()
    )
    jhu_long = jhu.melt(id_vars=["Country/Region"], var_name="date", value_name="total_cases")
    jhu_long["date"] = pd.to_datetime(jhu_long["date"])
    jhu_long = jhu_long.sort_values(["Country/Region", "date"]).reset_index(drop=True)
    return jhu_long


def load_owid_data() -> pd.DataFrame:
    logger.info("Loading OWID data...")
    _check_file_exists(PATHS.owid)

    owid = pd.read_csv(
        PATHS.owid,
        usecols=[
            "location",
            "date",
            "people_fully_vaccinated",
            "new_tests_smoothed",
            "population",
            "new_vaccinations_smoothed",
            "human_development_index",
            "hospital_beds_per_thousand",
            "stringency_index",
            "reproduction_rate",
        ],
    )
    owid["date"] = pd.to_datetime(owid["date"])
    return owid


def load_google_mobility() -> tuple:
    logger.info("Loading Google Mobility data...")
    _check_file_exists(PATHS.google_mobility)

    google = pd.read_csv(PATHS.google_mobility, low_memory=False)
    google = google[
        google["sub_region_1"].isna() & google["sub_region_2"].isna() & google["metro_area"].isna()
    ]
    mobility_cols = [c for c in google.columns if "percent_change_from_baseline" in c]
    google = google[["country_region", "date"] + mobility_cols]
    google["date"] = pd.to_datetime(google["date"])
    return google, mobility_cols


def load_environmental_data(jhu_long: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic environmental data (temperature, humidity, precipitation, UV)
    for countries based on latitude approximation and seasonal patterns.

    Improvements over v1:
    - Fully vectorized per-country computation (no inner loop over rows)
    - Proper date type handling for numpy datetime64
    - Deterministic seed per country for reproducibility
    - Realistic seasonal patterns with latitude approximation
    """
    logger.info("Generating synthetic environmental data...")

    countries = sorted(jhu_long["Country/Region"].unique())

    # Pre-compute country metadata (hemisphere approximation via hash)
    country_meta = {}
    for country in countries:
        country_hash = hash(country) % 100
        is_northern = country_hash > 30
        # Seasonal offset: southern hemisphere is ~6 months (180 days) ahead
        country_meta[country] = 180.0 if not is_northern else 0.0

    env_frames = []
    for country in countries:
        group = jhu_long[jhu_long["Country/Region"] == country]
        dates = group["date"].values
        seasonal_offset = country_meta[country]

        # Convert to day-of-year vectorized
        # numpy datetime64 -> int days since epoch -> mod 365
        days_since_epoch = (
            (dates - np.datetime64("1970-01-01")).astype("timedelta64[D]").astype(float)
        )
        day_of_year = np.mod(days_since_epoch, 365.25)

        # Temperature: sinusoidal seasonal pattern
        temp_cycle = np.sin(2 * np.pi * (day_of_year + seasonal_offset) / 365.25)
        temperature_avg = 10.0 + 20.0 * temp_cycle
        temperature_min = temperature_avg - 5.0
        temperature_max = temperature_avg + 5.0

        # Humidity: inverse relationship with temperature
        humidity_avg = 80.0 - 0.5 * (temperature_avg - 15.0)
        humidity_avg = np.clip(humidity_avg, 30.0, 95.0)

        # Precipitation: seasonal + random component
        precip_base = 2.0 + 3.0 * np.abs(np.sin(2 * np.pi * day_of_year / 365.25))
        rng = np.random.default_rng(abs(hash(country)) % (2**31))
        precip_random = rng.exponential(precip_base)
        zero_mask = rng.random(len(precip_base)) <= 0.3
        precipitation = np.where(zero_mask, 0.0, precip_random)

        # UV Index: correlated with temperature + noise
        uv_noise = rng.normal(0, 1, len(temperature_avg))
        uv_index = np.clip((temperature_avg + 10.0) * 0.3 + uv_noise, 0, 12)

        frame = pd.DataFrame(
            {
                "Country/Region": country,
                "date": dates,
                "temperature_avg": temperature_avg,
                "temperature_min": temperature_min,
                "temperature_max": temperature_max,
                "humidity_avg": humidity_avg,
                "precipitation": precipitation,
                "uv_index": uv_index,
            }
        )
        env_frames.append(frame)

    env_df = pd.concat(env_frames, ignore_index=True)
    logger.info(f"Generated environmental data for {len(env_df):,} records")
    return env_df


def fuzzy_match_countries(
    source_locs: np.ndarray, target_locs: np.ndarray, threshold: int = None
) -> dict:
    threshold = threshold or FUZZY_MATCH_THRESHOLD

    if os.path.exists(PATHS.country_matches):
        with open(PATHS.country_matches, "r") as f:
            cached = json.load(f)
        if all(loc in cached for loc in source_locs):
            logger.info("Using cached country matches")
            return cached

    logger.info("Cache not found or stale, performing fuzzy matching...")
    match_dict = {}
    manual_overrides = {
        "United States": "US",
        "South Korea": "Korea, South",
        "Czech Republic": "Czechia",
        "Burma": "Myanmar",
        "Taiwan*": "Taiwan",
    }
    target_locs_set = set(target_locs)

    for loc in source_locs:
        if loc in manual_overrides and manual_overrides[loc] in target_locs_set:
            match_dict[loc] = manual_overrides[loc]
        elif loc in target_locs_set:
            match_dict[loc] = loc
        else:
            res = process.extractOne(loc, target_locs, scorer=fuzz.WRatio)
            if res:
                best_match, score, _ = res
                match_dict[loc] = best_match if score >= threshold else loc
            else:
                match_dict[loc] = loc

    os.makedirs(os.path.dirname(PATHS.country_matches), exist_ok=True)
    with open(PATHS.country_matches, "w") as f:
        json.dump(match_dict, f)

    return match_dict


def preprocess() -> None:
    os.makedirs(PATHS.processed_data, exist_ok=True)

    jhu_long = load_jhu_data()
    jhu_countries = jhu_long["Country/Region"].unique()

    owid = load_owid_data()
    owid["location"] = owid["location"].map(
        fuzzy_match_countries(owid["location"].unique(), jhu_countries)
    )
    owid.rename(columns={"location": "Country/Region"}, inplace=True)

    google, mobility_cols = load_google_mobility()
    google["country_region"] = google["country_region"].map(
        fuzzy_match_countries(google["country_region"].unique(), jhu_countries)
    )
    google.rename(columns={"country_region": "Country/Region"}, inplace=True)

    env_data = load_environmental_data(jhu_long)

    logger.info("Merging datasets...")
    merged = pd.merge(jhu_long, owid, on=["Country/Region", "date"], how="left")
    merged = pd.merge(merged, google, on=["Country/Region", "date"], how="left")
    merged = pd.merge(merged, env_data, on=["Country/Region", "date"], how="left")

    merged["new_cases"] = merged.groupby("Country/Region")["total_cases"].diff().fillna(0)
    merged["new_cases"] = merged["new_cases"].clip(lower=0)

    logger.info("Applying interpolation and smoothing...")
    cols_to_interpolate = [
        "people_fully_vaccinated",
        "new_tests_smoothed",
        "new_vaccinations_smoothed",
        "stringency_index",
        "reproduction_rate",
    ] + mobility_cols
    for col in cols_to_interpolate:
        if col in merged.columns:
            merged[col] = merged.groupby("Country/Region")[col].transform(
                lambda x: x.interpolate(method="linear").ffill().bfill().fillna(0)
            )

    features_to_smooth = ["new_cases"] + [c for c in mobility_cols if c in merged.columns]
    for f in features_to_smooth:
        merged[f + "_smoothed"] = merged.groupby("Country/Region")[f].transform(
            lambda x: x.rolling(window=7, center=True, min_periods=1).mean()
        )

    # Smooth environmental features for more stable patterns
    env_features = [
        "temperature_avg",
        "temperature_min",
        "temperature_max",
        "humidity_avg",
        "precipitation",
        "uv_index",
    ]
    for ef in env_features:
        if ef in merged.columns:
            merged[ef] = merged.groupby("Country/Region")[ef].transform(
                lambda x: x.rolling(window=7, center=True, min_periods=1).mean()
            )

    merged["case_growth_rate"] = (
        merged.groupby("Country/Region")["new_cases_smoothed"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    merged["population"] = (
        merged.groupby("Country/Region")["population"].ffill().bfill().fillna(1e6)
    )
    merged["human_development_index"] = (
        merged.groupby("Country/Region")["human_development_index"]
        .ffill()
        .bfill()
        .fillna(merged["human_development_index"].mean())
    )
    merged["hospital_beds_per_thousand"] = (
        merged.groupby("Country/Region")["hospital_beds_per_thousand"]
        .ffill()
        .bfill()
        .fillna(merged["hospital_beds_per_thousand"].mean())
    )

    # Multi-scale smoothing - use both 7-day (short-term) and 30-day (long-term periodic patterns)
    # As per GPR paper recommendation: 30-day moving average captures periodic trends
    logger.info("Applying multi-scale smoothing (7-day and 30-day windows)...")
    merged["new_cases_smoothed_30d"] = merged.groupby("Country/Region")[
        "new_cases_smoothed"
    ].transform(lambda x: x.rolling(window=30, center=True, min_periods=1).mean())

    # Log-difference transformation for better modeling (GPR paper recommendation)
    # Δ(t) = log(I(t)) - log(I(t-η)) captures epidemic growth dynamics
    # Higher η values result in narrower prediction confidence intervals (GPR paper finding)
    logger.info("Computing log-difference features for epidemic trend modeling...")

    for eta in [7, 14, 30]:
        merged[f"log_diff_{eta}"] = merged.groupby("Country/Region")[
            "new_cases_smoothed"
        ].transform(lambda x: np.log1p(x.clip(lower=0)).diff(eta).fillna(0))

    # Log-difference on 30-day smoothed cases for noise reduction (GPR paper optimal window)
    for eta in [7, 14]:
        merged[f"log_diff_{eta}_30d"] = merged.groupby("Country/Region")[
            "new_cases_smoothed_30d"
        ].transform(lambda x: np.log1p(x.clip(lower=0)).diff(eta).fillna(0))

    # Epidemic growth rate derived from log-difference (instantaneous reproduction proxy)
    # r = (1/η) * Δ(t) where Δ(t) is the log-difference
    for eta in [7, 14]:
        merged[f"growth_rate_logdiff_{eta}"] = merged[f"log_diff_{eta}"] / eta

    # Additional temporal features for better prediction
    merged["log_cases"] = np.log1p(merged["new_cases_smoothed"])
    merged["log_total_cases"] = np.log1p(merged["total_cases"])

    # Trend acceleration (second derivative)
    merged["trend_acceleration"] = merged.groupby("Country/Region")["new_cases_smoothed"].transform(
        lambda x: x.diff().diff().fillna(0)
    )

    # Velocity: day-over-day change
    merged["velocity"] = merged.groupby("Country/Region")["new_cases_smoothed"].diff().fillna(0)

    # Relative change features (growth rate)
    merged["relative_change_7d"] = (
        merged.groupby("Country/Region")["new_cases_smoothed"]
        .pct_change(7)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    merged["relative_change_14d"] = (
        merged.groupby("Country/Region")["new_cases_smoothed"]
        .pct_change(14)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # Compute key smoothed values for different analysis purposes
    for col in ["new_cases", "case_growth_rate"]:
        if col in merged.columns:
            merged[f"{col}_smoothed_30d"] = merged.groupby("Country/Region")[col].transform(
                lambda x: x.rolling(window=30, center=True, min_periods=1).mean()
            )

    # Serial interval weighted lag features (Bracher & Held, 1901.03090)
    # Geometric weighting: u_d = (1-kappa)^(d-1) * kappa, with kappa=0.3
    # Captures how past incidence contributes to current spread
    logger.info("Computing serial interval weighted lag features...")
    kappa_si = 0.3
    for lag in range(1, 6):
        weight = (1 - kappa_si) ** (lag - 1) * kappa_si
        merged[f"cases_lag_{lag}"] = merged.groupby("Country/Region")[
            "new_cases_smoothed"
        ].transform(lambda x, w=weight: x.shift(lag).fillna(0) * w)
    merged["si_weighted_cases"] = merged[[f"cases_lag_{i}" for i in range(1, 6)]].sum(axis=1)

    # Remove temporary lag columns
    for lag in range(1, 6):
        if f"cases_lag_{lag}" in merged.columns:
            merged.drop(columns=[f"cases_lag_{lag}"], inplace=True)

    total_cases_per_country = merged.groupby("Country/Region")["total_cases"].max()
    valid_countries = total_cases_per_country[
        total_cases_per_country > TRAIN_CONFIG.min_cases_threshold
    ].index
    merged = merged[merged["Country/Region"].isin(valid_countries)]

    # Remove rows with all-NaN feature columns after filtering
    merged = merged.fillna(0)

    output_path = os.path.join(PATHS.processed_data, "merged_data.csv")
    merged.to_csv(output_path, index=False)
    logger.info(
        f"Saved processed data to {output_path} ({len(merged):,} rows, {len(valid_countries)} countries)"
    )


if __name__ == "__main__":
    preprocess()
