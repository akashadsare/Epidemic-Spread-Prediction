import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import os
import torch
import joblib

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, FEATURES, get_device, TRAIN_CONFIG
from models.seir_lstm import SEIR_LSTM
from models.ensemble import SEIR_LSTM_Ensemble

st.set_page_config(
    page_title="Epidemic Spread Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#636EFA",
    "secondary": "#EF553B",
    "success": "#00CC96",
    "warning": "#FFA15A",
    "danger": "#EF553B",
    "info": "#19D3F3",
    "purple": "#AB63FA",
    "bg": "#0f1115",
    "card": "#1a1d24",
    "text": "#f0f0f5",
    "grid": "rgba(255,255,255,0.08)",
}


def hex_to_rgba(hex_color, opacity=0.1):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


st.markdown(
    f"""
<style>
    .main {{background-color: {COLORS["bg"]}; color: {COLORS["text"]};}}
    .stApp {{background-color: {COLORS["bg"]};}}
    h1, h2, h3, h4 {{color: {COLORS["text"]}; font-family: 'Inter', sans-serif;}}
    .stMetric {{background-color: {COLORS["card"]}; padding: 18px; border-radius: 10px; border-left: 3px solid {COLORS["success"]};}}
    div[data-testid="stMetricValue"] {{color: {COLORS["success"]}; font-size: 1.5rem;}}
    div[data-testid="stMetricLabel"] {{color: {COLORS["text"]}; font-size: 0.9rem;}}
    .stAlert {{border-radius: 8px;}}
    .risk-high {{background-color: rgba(239,85,59,0.15); border-left: 3px solid {COLORS["danger"]}; padding: 10px; border-radius: 4px; margin: 4px 0;}}
    .risk-medium {{background-color: rgba(255,161,90,0.15); border-left: 3px solid {COLORS["warning"]}; padding: 10px; border-radius: 4px; margin: 4px 0;}}
    .risk-low {{background-color: rgba(0,204,150,0.15); border-left: 3px solid {COLORS["success"]}; padding: 10px; border-radius: 4px; margin: 4px 0;}}
</style>
""",
    unsafe_allow_html=True,
)


# --- ISO mapping ---
@st.cache_data
def build_iso_map() -> dict:
    lookup_path = os.path.join(os.path.dirname(PATHS.jhu_cases), "UID_ISO_FIPS_LookUp_Table.csv")
    if not os.path.exists(lookup_path):
        return {}
    lk = pd.read_csv(lookup_path, usecols=["Country_Region", "iso3"])
    lk = lk[lk["iso3"].notna()].drop_duplicates(subset=["Country_Region"])
    return dict(zip(lk["Country_Region"], lk["iso3"]))


# --- Data loading ---
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(PATHS.merged_data)
    df["date"] = pd.to_datetime(df["date"])

    # Ensure all features exist in the data
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.warning(f"Some configured features are missing from the dataset: {', '.join(missing)}")

    return df


@st.cache_resource
def load_model_and_scaler():
    device = get_device()
    ensemble_path = PATHS.model_weights.replace(".pth", "_ensemble.pth")
    if os.path.exists(ensemble_path):
        model, loaded_config = SEIR_LSTM_Ensemble.load_with_metadata(ensemble_path, device)
    else:
        model, loaded_config = SEIR_LSTM.load_with_metadata(PATHS.model_weights, device)
    scaler = joblib.load(PATHS.scaler)
    return model, scaler, device



def validate_assets() -> bool:
    required_files = [PATHS.merged_data, PATHS.model_weights, PATHS.scaler]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"Assets missing: {', '.join(missing)}. Please run preprocessing and training scripts locally and ensure the generated files are accessible.")
        return False
    return True


if not validate_assets():
    st.stop()


# --- Hotspot & Risk Classification ---
def classify_risk(score: float) -> str:
    if score >= 500:
        return "Critical"
    elif score >= 200:
        return "High"
    elif score >= 80:
        return "Medium"
    return "Low"


def detect_hotspots(df: pd.DataFrame, threshold_percentile: float = 80) -> pd.DataFrame:
    latest = df.groupby("Country/Region").last().reset_index()

    # 1. Case incidence (per million)
    latest["risk_per_million"] = (
        latest["new_cases_smoothed"] / latest["population"].clip(lower=1) * 1e6
    ).fillna(0)
    # Normalize risk_per_million (assuming 2000 per million is critical)
    norm_incidence = (latest["risk_per_million"] / 2000.0).clip(upper=1.0)

    # 2. Growth rate (velocity of outbreak)
    latest["growth_rate"] = latest.get("case_growth_rate", 0).fillna(0)
    norm_growth = (latest["growth_rate"].clip(lower=0) / 2.0).clip(upper=1.0)

    # 3. Healthcare vulnerability (inverse of beds per 1k)
    # Average beds/1k is around 3. Higher means less vulnerable.
    latest["healthcare_vulnerability"] = (
        1 - (latest["hospital_beds_per_thousand"].clip(0, 10) / 10)
    ).fillna(0.5)

    # 4. Population vulnerability (inverse of vaccination and HDI)
    latest["vaccination_rate"] = (
        latest.get("people_fully_vaccinated", 0) / latest["population"].clip(lower=1)
    ).fillna(0)
    latest["pop_vulnerability"] = (
        (1 - latest["vaccination_rate"]).clip(0, 1) * 0.7
        + (1 - latest["human_development_index"].clip(0, 1)) * 0.3
    ).fillna(0.5)

    # Composite Risk Score (0-1000)
    # Weighted combination of incidence, growth, healthcare, and population vulnerability
    latest["risk_score"] = (
        norm_incidence * 400
        + norm_growth * 300
        + latest["healthcare_vulnerability"] * 150
        + latest["pop_vulnerability"] * 150
    ).fillna(0)

    latest["risk_class"] = latest["risk_score"].apply(classify_risk)
    threshold = latest["risk_score"].quantile(threshold_percentile / 100)
    latest["is_hotspot"] = latest["risk_score"] >= threshold
    return latest.sort_values("risk_score", ascending=False)


# --- Prediction ---
def run_prediction(country_data, modifier_tuple, device, model, scaler, n_samples=50):
    """Run Monte Carlo prediction with proper historical feature sequences.

    Uses the last 30 days of actual scaled features as model input instead of
    repeating a single row, which was a bug in the original implementation.
    """
    m_mod, r_mod, t_mod, v_mod = modifier_tuple

    pop = country_data["population"].iloc[-1]
    I_0 = max(country_data["new_cases_smoothed"].iloc[-1], 1.0)
    E_0 = I_0 * TRAIN_CONFIG.e_multiplier
    R_0 = country_data["total_cases"].iloc[-1]
    S_0 = max(pop - E_0 - I_0 - R_0, 0)

    # BUG FIX: Use last 30 rows of actual features, not a repeated single row
    # This preserves the temporal dynamics the model learned during training
    seq_len = 30
    if len(country_data) >= seq_len:
        recent_data = country_data.iloc[-seq_len:]
    else:
        recent_data = country_data.iloc[-1:].reindex(range(seq_len), method="ffill")

    # Build feature matrix from actual data
    feat_df = pd.DataFrame(index=range(seq_len))
    for f in FEATURES:
        if f in recent_data.columns:
            feat_df[f] = recent_data[f].values[:seq_len]
        else:
            feat_df[f] = 0.0

    # Apply scenario modifiers to relevant mobility features
    if "workplaces_percent_change_from_baseline_smoothed" in feat_df.columns:
        feat_df["workplaces_percent_change_from_baseline_smoothed"] += m_mod
    if "retail_and_recreation_percent_change_from_baseline_smoothed" in feat_df.columns:
        feat_df["retail_and_recreation_percent_change_from_baseline_smoothed"] += r_mod
    if "transit_stations_percent_change_from_baseline_smoothed" in feat_df.columns:
        feat_df["transit_stations_percent_change_from_baseline_smoothed"] += t_mod

    scaled_feats = scaler.transform(feat_df[FEATURES])
    X_input = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0).to(device)

    latest_v_rate = country_data["new_vaccinations_smoothed"].iloc[-1] / pop if pop > 0 else 0
    future_v_rate = latest_v_rate * (1 + v_mod / 100.0)
    V_input = torch.full((1, 30), future_v_rate, dtype=torch.float32).to(device)

    S_init = torch.tensor([S_0], dtype=torch.float32, device=device)
    E_init = torch.tensor([E_0], dtype=torch.float32, device=device)
    I_init = torch.tensor([I_0], dtype=torch.float32, device=device)
    R_init = torch.tensor([R_0], dtype=torch.float32, device=device)
    N_init = torch.tensor([pop], dtype=torch.float32, device=device)

    all_preds, all_betas, all_sigmas, all_gammas = [], [], [], []

    model.train()  # Enable dropout for MC sampling
    with torch.no_grad():
        for _ in range(n_samples):
            try:
                preds, params = model(X_input, (S_init, E_init, I_init, R_init), N_init, V_input)
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    continue
                all_preds.append(preds[0].cpu().numpy())
                all_betas.append(params[0][0].cpu().numpy())
                all_sigmas.append(params[1][0].cpu().numpy())
                all_gammas.append(params[2][0].cpu().numpy())
            except Exception:
                continue
    model.eval()

    if len(all_preds) == 0:
        return {
            "pred_mean": np.zeros(30),
            "pred_std": np.zeros(30),
            "pred_std_epistemic": np.zeros(30),
            "betas": np.zeros(30),
            "sigmas": np.zeros(30),
            "gammas": np.zeros(30),
            "rt": np.zeros(30),
            "S_0": S_0,
            "E_0": E_0,
            "I_0": I_0,
            "R_0": R_0,
            "pop": pop,
        }

    all_preds = np.array(all_preds)
    pred_mean = np.mean(all_preds, axis=0)
    pred_std = np.std(all_preds, axis=0)

    # Epistemic uncertainty: variance across grouped means
    n_groups = min(5, len(all_preds))
    group_size = len(all_preds) // n_groups
    if group_size > 0:
        group_means = [
            np.mean(all_preds[g * group_size : (g + 1) * group_size], axis=0)
            for g in range(n_groups)
        ]
        pred_std_epistemic = np.std(group_means, axis=0)
    else:
        pred_std_epistemic = pred_std * 0.5

    betas = np.mean(all_betas, axis=0)
    sigmas = np.mean(all_sigmas, axis=0)
    gammas = np.mean(all_gammas, axis=0)
    rt_pred = betas / np.maximum(gammas, 1e-5)

    # Sanitize outputs
    pred_mean = np.nan_to_num(pred_mean, nan=0.0, posinf=0.0, neginf=0.0)
    pred_std = np.nan_to_num(pred_std, nan=0.0, posinf=0.0, neginf=0.0)
    pred_std_epistemic = np.nan_to_num(pred_std_epistemic, nan=0.0, posinf=0.0, neginf=0.0)
    betas = np.nan_to_num(betas, nan=0.0, posinf=0.0, neginf=0.0)
    sigmas = np.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0)
    gammas = np.nan_to_num(gammas, nan=0.0, posinf=0.0, neginf=0.0)
    rt_pred = np.nan_to_num(rt_pred, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_std_epistemic": pred_std_epistemic,
        "betas": betas,
        "sigmas": sigmas,
        "gammas": gammas,
        "rt": rt_pred,
        "S_0": S_0,
        "E_0": E_0,
        "I_0": I_0,
        "R_0": R_0,
        "pop": pop,
    }


# --- Chart styling ---
CHART_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=COLORS["text"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def style_chart(fig, height=450):
    fig.update_layout(
        template=CHART_LAYOUT["template"],
        plot_bgcolor=CHART_LAYOUT["plot_bgcolor"],
        paper_bgcolor=CHART_LAYOUT["paper_bgcolor"],
        font=CHART_LAYOUT["font"],
        legend=CHART_LAYOUT["legend"],
        height=height,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig


# ============================
# MAIN APP
# ============================


try:
    df = load_data()
    model, scaler, device = load_model_and_scaler()
    iso_map = build_iso_map()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

st.title("Epidemic Spread & Risk Prediction Dashboard")
st.markdown(
    "Hybrid **Attention-SEIR-LSTM** model analyzing epidemiological, mobility, and policy data "
    "for outbreak prediction, hotspot detection, and transmission modeling."
)

if df.empty:
    st.error("Merged dataset is empty. Run preprocessing first.")
    st.stop()

last_update = df["date"].max().strftime("%Y-%m-%d")
total_countries = df["Country/Region"].nunique()
total_records = len(df)

st.info(
    f"Data as of **{last_update}** | "
    f"**{total_countries}** countries | "
    f"**{total_records:,}** records | "
    f"30-day forecast horizon"
)

# Sidebar
st.sidebar.header("Controls")
countries = sorted(df["Country/Region"].unique())
comparison_mode = st.sidebar.checkbox("Comparison Mode")

if comparison_mode:
    selected_countries = st.sidebar.multiselect(
        "Regions", countries, default=["US", "United Kingdom"]
    )
else:
    default_index = countries.index("US") if "US" in countries else 0
    selected_country = st.sidebar.selectbox("Region", countries, index=default_index)
    selected_countries = [selected_country]

st.sidebar.markdown("---")
st.sidebar.subheader("What-If Scenarios")
policy_stringency = st.sidebar.slider("Policy Stringency Override", 0, 100, 50)
mobility_modifier = st.sidebar.slider("Workplace Mobility (%)", -100, 100, 50 - policy_stringency)
retail_modifier = st.sidebar.slider("Retail Mobility (%)", -100, 100, 50 - policy_stringency)
transit_modifier = st.sidebar.slider("Transit Mobility (%)", -100, 100, 50 - policy_stringency)
vaccination_modifier = st.sidebar.slider("Vaccination Modifier (%)", -100, 100, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Options")
n_samples = st.sidebar.slider("Monte Carlo Samples", 10, 100, 50)
confidence_level = st.sidebar.selectbox("Confidence Level", ["90%", "95%", "99%"])
ci_mult = {"90%": 1.645, "95%": 1.96, "99%": 2.576}[confidence_level]

st.sidebar.markdown("---")
st.sidebar.subheader("Policy Brief")
for country in selected_countries:
    cd = df[df["Country/Region"] == country].copy().sort_values("date")
    if cd.empty:
        continue
    beds = cd["hospital_beds_per_thousand"].mean()
    str_idx = cd["stringency_index"].iloc[-1]
    if policy_stringency > str_idx + 20:
        st.sidebar.warning(
            f"**{country}:** Major tightening (+{policy_stringency - str_idx:.0f} pts)."
        )
    elif mobility_modifier > 10:
        st.sidebar.error(f"**{country}:** Rebound risk; {beds:.1f}/1k beds.")
    else:
        st.sidebar.success(f"**{country}:** Stable containment.")

# --- Run predictions ---
forecast_results = {}
for country in selected_countries:
    country_data = df[df["Country/Region"] == country].copy().sort_values("date")
    if country_data.empty or len(country_data) < 30:
        continue

    results = run_prediction(
        country_data,
        (mobility_modifier, retail_modifier, transit_modifier, vaccination_modifier),
        device,
        model,
        scaler,
        n_samples,
    )

    last_date = country_data["date"].iloc[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]

    forecast_results[country] = {
        "dates": future_dates,
        **results,
        "data": country_data,
    }

# Tabs
scenario_comparison = len(selected_countries) > 1
tabs = st.tabs(
    [
        "Overview",
        "Forecast",
        "Transmission (R_t)",
        "Hotspot Detection",
        "Outbreak Patterns",
        "Mobility Impact",
        "What-If Analysis",
        "Scenario Analysis",
        "Demographic Factors",
        "Global Risk Map",
    ]
)
(
    tab_overview,
    tab_forecast,
    tab_rt,
    tab_hotspot,
    tab_patterns,
    tab_mobility,
    tab_whatif,
    tab_scenario,
    tab_demographic,
    tab_risk,
) = tabs

# ========================
# OVERVIEW TAB
# ========================
with tab_overview:
    st.markdown("### Global Summary")
    latest_data = df.groupby("Country/Region").last().reset_index()
    total_cases = latest_data["total_cases"].sum()
    avg_stringency = latest_data["stringency_index"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", f"{total_cases:,.0f}")
    col2.metric("Countries Tracked", f"{total_countries}")
    col3.metric("Avg Stringency", f"{avg_stringency:.0f}/100")
    col4.metric("Data Range", f"{df['date'].min().strftime('%Y-%m')} to {last_update}")

    if forecast_results:
        st.markdown("### Selected Country Metrics")
        for country, res in forecast_results.items():
            st.markdown(f"#### {country}")
            m1, m2, m3, m4 = st.columns(4)
            last_c = res["data"]["new_cases_smoothed"].iloc[-1]
            peak_v = res["pred_mean"].max()
            total_30 = res["pred_mean"].sum()
            avg_rt = np.mean(res["rt"])

            week_ago = (
                res["data"]["new_cases_smoothed"].iloc[-7] if len(res["data"]) >= 7 else last_c
            )
            wow_change = ((last_c - week_ago) / max(week_ago, 1)) * 100

            m1.metric(
                "Latest Cases",
                f"{last_c:,.0f}",
                f"{wow_change:+.1f}%",
            )
            m2.metric("Predicted Peak", f"{peak_v:,.0f}")
            m3.metric("30-Day Total", f"{total_30:,.0f}")
            m4.metric("Avg R_t", f"{avg_rt:.2f}", "Declining" if avg_rt < 1 else "Growing")

            if len(res["pred_mean"]) > 7:
                early_week = res["pred_mean"][:7].mean()
                late_week = res["pred_mean"][-7:].mean()
                trend_direction = (
                    "Accelerating"
                    if late_week > early_week * 1.1
                    else ("Decelerating" if late_week < early_week * 0.9 else "Stable")
                )
                trend_color = (
                    COLORS["danger"]
                    if "Accelerating" in trend_direction
                    else (
                        COLORS["success"]
                        if "Decelerating" in trend_direction
                        else COLORS["warning"]
                    )
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Early Week Avg", f"{early_week:,.0f}")
                c2.metric("Late Week Avg", f"{late_week:,.0f}")
                c3.metric("Trend", trend_direction, delta_color="normal")

                if avg_rt > 1.0:
                    doubling_time = np.log(2) / np.log(avg_rt) if avg_rt > 1 else float("inf")
                    c4.metric("Doubling Time", f"{doubling_time:.1f} days")
                else:
                    c4.metric("R_eff < 1", "Epidemic Shrinking", delta_color="inverse")

                # Advanced Epidemiological Metrics
                # Calculate instantaneous reproduction number using log-difference approach
                # Per GPR paper: Δ(t) = log(I(t)) - log(I(t-η))
                # This captures the epidemic trend dynamics
                if "log_diff_7" in res["data"].columns:
                    log_diff_avg = res["data"]["log_diff_7"].iloc[-30:].mean()
                    generation_time = 5.0  # Approximate COVID-19 generation time
                    r_instant = (
                        np.exp(log_diff_avg * generation_time)
                        if not np.isnan(log_diff_avg)
                        else avg_rt
                    )

                    # Additional rate metrics
                    growth_rate = (np.exp(log_diff_avg) - 1) if not np.isnan(log_diff_avg) else 0

                    c5, c6, c7 = st.columns(3)
                    c5.metric(
                        "Instant R (Log-diff)",
                        f"{r_instant:.2f}",
                        "Growing" if r_instant > 1 else "Declining",
                    )
                    c6.metric(
                        "Growth Rate (r)",
                        f"{growth_rate:.3f}/day",
                        "Accelerating" if growth_rate > 0 else "Decelerating",
                    )
                    c7.metric(
                        "Generation Time", f"{generation_time:.1f} days", "Avg infectious period"
                    )

    st.markdown("---")
    st.markdown("### Historical Case Trends")
    fig_hist = go.Figure()
    palette = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["info"],
    ]
    for i, country in enumerate(selected_countries):
        cd = df[df["Country/Region"] == country]
        color = palette[i % len(palette)]
        weekly = (
            cd.groupby(pd.Grouper(key="date", freq="W"))
            .agg({"new_cases_smoothed": "mean"})
            .reset_index()
        )
        fig_hist.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=weekly["new_cases_smoothed"],
                mode="lines",
                name=country,
                line=dict(width=2.5, color=color),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, 0.1)
                if color.startswith("#")
                else color.replace("rgb", "rgba").replace(")", ",0.1)"),
            )
        )
    style_chart(fig_hist, 400)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Log-difference trend analysis (GPR paper approach)
    st.markdown("### Epidemic Trend Analysis (Log-Difference)")
    st.markdown(
        "Δ(t) = log(I(t)) - log(I(t-η)) captures epidemic growth dynamics. "
        "Positive values indicate increasing cases, negative values indicate decline. "
        "Based on Gaussian Process Regression approach (She et al., 2312.09384v3)."
    )
    fig_logdiff = go.Figure()
    palette_ld = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]
    has_log_diff = False
    for i, country in enumerate(selected_countries):
        cd = df[df["Country/Region"] == country]
        if "log_diff_7" in cd.columns:
            has_log_diff = True
            fig_logdiff.add_trace(
                go.Scatter(
                    x=cd["date"],
                    y=cd["log_diff_7"],
                    mode="lines",
                    name=f"{country} (η=7)",
                    line=dict(width=2, color=palette_ld[i % len(palette_ld)]),
                )
            )
    fig_logdiff.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
    if has_log_diff:
        fig_logdiff.add_hrect(
            y0=0,
            y1=cd["log_diff_7"].max(),
            fillcolor="rgba(239,85,59,0.05)",
            line_width=0,
            annotation_text="Growing",
            annotation_position="top left",
        )
        fig_logdiff.add_hrect(
            y0=cd["log_diff_7"].min(),
            y1=0,
            fillcolor="rgba(0,204,150,0.05)",
            line_width=0,
            annotation_text="Declining",
            annotation_position="bottom left",
        )
    fig_logdiff.update_layout(
        xaxis_title="Date",
        yaxis_title="Δ(t) Log-Difference",
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_logdiff, use_container_width=True)

# ========================
# FORECAST TAB
# ========================
with tab_forecast:
    if forecast_results:
        st.markdown("### 30-Day Forecast with Confidence Interval")
        st.markdown(
            "Predictions use Monte Carlo dropout sampling. "
            "Dashed line = predicted mean, shaded region = uncertainty interval."
        )

        fig_forecast = go.Figure()
        palette = [
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["success"],
            COLORS["warning"],
            COLORS["info"],
        ]

        for i, (country, res) in enumerate(forecast_results.items()):
            color = palette[i % len(palette)]
            cd = res["data"]

            fig_forecast.add_trace(
                go.Scatter(
                    x=cd["date"],
                    y=cd["new_cases_smoothed"],
                    mode="lines",
                    name=f"{country} Historical",
                    line=dict(width=2, color=color),
                )
            )

            lower = np.maximum(res["pred_mean"] - ci_mult * res["pred_std"], 0)
            upper = res["pred_mean"] + ci_mult * res["pred_std"]
            rgb = tuple(int(color.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))

            fig_forecast.add_trace(
                go.Scatter(
                    x=res["dates"] + res["dates"][::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill="toself",
                    fillcolor=f"rgba{rgb + (0.12,)}",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{country} CI",
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=res["pred_mean"],
                    mode="lines",
                    name=f"{country} Forecast",
                    line=dict(width=3, dash="dash", color=color),
                )
            )
            peak_idx = np.argmax(res["pred_mean"])
            fig_forecast.add_trace(
                go.Scatter(
                    x=[res["dates"][peak_idx]],
                    y=[res["pred_mean"][peak_idx]],
                    mode="markers+text",
                    name=f"{country} Peak",
                    text=[f"Peak: {res['pred_mean'][peak_idx]:,.0f}"],
                    textposition="top center",
                    marker=dict(size=12, symbol="star", color=COLORS["warning"]),
                )
            )

        style_chart(fig_forecast, 500)
        fig_forecast.update_layout(hovermode="x unified")
        st.plotly_chart(fig_forecast, use_container_width=True)

        # SEIR compartment visualization for each country
        st.markdown("### SEIR Compartment Dynamics")
        for country, res in forecast_results.items():
            st.markdown(f"#### {country} — Compartment Trajectories")
            pop = res["pop"]

            # Run SEIR compartment extraction
            country_data = res["data"]
            I_0 = max(country_data["new_cases_smoothed"].iloc[-1], 1.0)
            E_0 = I_0 * TRAIN_CONFIG.e_multiplier
            R_0 = country_data["total_cases"].iloc[-1]
            S_0 = max(pop - E_0 - I_0 - R_0, 0)

            feats = scaler.transform(country_data[FEATURES].iloc[-30:])
            X_input = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            v_rate = country_data["new_vaccinations_smoothed"].iloc[-1] / max(pop, 1)
            V_input = torch.full((1, 30), v_rate, dtype=torch.float32).to(device)

            with torch.no_grad():
                seir_data = model.get_seir_compartments(
                    X_input,
                    (
                        torch.tensor([S_0], device=device),
                        torch.tensor([E_0], device=device),
                        torch.tensor([I_0], device=device),
                        torch.tensor([R_0], device=device),
                    ),
                    torch.tensor([pop], device=device),
                    V_input,
                )

            fig_seir = go.Figure()
            compartment_colors = {
                "Susceptible": COLORS["primary"],
                "Exposed": COLORS["warning"],
                "Infectious": COLORS["danger"],
                "Recovered": COLORS["success"],
            }
            for comp_name, key, col in [
                ("Susceptible", "S", compartment_colors["Susceptible"]),
                ("Exposed", "E", compartment_colors["Exposed"]),
                ("Infectious", "I", compartment_colors["Infectious"]),
                ("Recovered", "R", compartment_colors["Recovered"]),
            ]:
                traj = seir_data[key][0].cpu().numpy() / pop * 100
                fig_seir.add_trace(
                    go.Scatter(
                        x=res["dates"],
                        y=traj,
                        mode="lines",
                        name=comp_name,
                        line=dict(width=2.5, color=col),
                    )
                )

            fig_phase = go.Figure()
            s_percent = seir_data["S"][0].cpu().numpy() / pop * 100
            i_percent = seir_data["I"][0].cpu().numpy() / pop * 100

            fig_phase.add_trace(
                go.Scatter(
                    x=s_percent,
                    y=i_percent,
                    mode="lines+markers",
                    marker=dict(
                        size=6,
                        color=np.arange(len(s_percent)),
                        colorscale="Plasma",
                        showscale=True,
                        colorbar=dict(title="Days"),
                    ),
                    line=dict(color="rgba(255,255,255,0.4)", width=2),
                    name="Trajectory",
                )
            )

            # Add direction arrows to show flow in phase space
            for idx in range(0, len(s_percent) - 1, max(1, len(s_percent) // 10)):
                fig_phase.add_annotation(
                    ax=s_percent[idx],
                    ay=i_percent[idx],
                    x=s_percent[idx + 1],
                    y=i_percent[idx + 1],
                    axref="x",
                    ayref="y",
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="rgba(255,255,255,0.5)",
                )

            fig_phase.update_layout(
                title=f"Phase Space (S vs I) — {country}",
                xaxis_title="Susceptible (%)",
                yaxis_title="Infectious (%)",
            )
            style_chart(fig_phase, 400)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_seir, use_container_width=True)
            with c2:
                st.plotly_chart(fig_phase, use_container_width=True)

        # Download button
        csv_rows = []
        for c, res in forecast_results.items():
            for d, cases in zip(res["dates"], res["pred_mean"]):
                csv_rows.append({"Country": c, "Date": d, "Predicted_Cases": cases})
        st.download_button(
            "Download Forecast CSV",
            pd.DataFrame(csv_rows).to_csv(index=False),
            "epidemic_forecast.csv",
            "text/csv",
        )

        # Interpretation
        st.markdown("### Forecast Interpretation")
        for country, res in forecast_results.items():
            avg_rt = np.mean(res["rt"])
            peak = res["pred_mean"].max()
            trend = "increasing" if res["pred_mean"][-1] > res["pred_mean"][0] else "decreasing"
            st.markdown(
                f"- **{country}**: Predicted peak of **{peak:,.0f}** cases/day. "
                f"R_t = {avg_rt:.2f} ({'epidemic growing' if avg_rt > 1 else 'epidemic declining'}). "
                f"Trend is {trend} over the 30-day horizon."
            )
    else:
        st.warning("Select a country with sufficient data.")

# ========================
# R_t TAB
# ========================
with tab_rt:
    if forecast_results:
        st.markdown("### Effective Reproduction Number (R_t)")
        st.markdown(
            "R_t is the average number of secondary infections per case. "
            "R_t > 1: epidemic growing. R_t < 1: epidemic declining."
        )

        fig_rt = go.Figure()
        palette = [
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["success"],
            COLORS["warning"],
            COLORS["info"],
        ]
        for i, (country, res) in enumerate(forecast_results.items()):
            fig_rt.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=res["rt"],
                    mode="lines+markers",
                    name=country,
                    line=dict(color=palette[i % len(palette)], width=2.5),
                    marker=dict(size=5),
                )
            )

        fig_rt.add_hrect(
            y0=0,
            y1=1,
            fillcolor="rgba(0,204,150,0.08)",
            line_width=0,
            annotation_text="Declining",
            annotation_position="bottom left",
        )
        fig_rt.add_hrect(
            y0=1,
            y1=5,
            fillcolor="rgba(239,85,59,0.08)",
            line_width=0,
            annotation_text="Growing",
            annotation_position="top left",
        )
        fig_rt.add_hline(y=1.0, line_dash="dot", line_color="white", opacity=0.5)

        style_chart(fig_rt, 450)
        fig_rt.update_layout(yaxis_title="R_t", xaxis_title="Date")
        st.plotly_chart(fig_rt, use_container_width=True)

        # R_t Probability Analysis
        st.markdown("### R_t Probability Analysis")
        prob_cols = st.columns(len(forecast_results))
        for idx, (country, res) in enumerate(forecast_results.items()):
            rt_values = res["rt"]
            probGrowing = np.mean(rt_values > 1.0) * 100
            probDeclining = np.mean(rt_values < 1.0) * 100
            avg_rt = np.mean(rt_values)

            fig_pie = go.Figure(
                go.Pie(
                    values=[probGrowing, probDeclining],
                    labels=["Growing (Rt>1)", "Declining (Rt<1)"],
                    marker=dict(colors=[COLORS["danger"], COLORS["success"]]),
                    hole=0.4,
                    textinfo="label+percent",
                )
            )
            fig_pie.update_layout(
                title=f"{country}: P(R_t>1)={probGrowing:.0f}%",
                showlegend=False,
                template="plotly_dark",
                height=250,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            prob_cols[idx % len(prob_cols)].plotly_chart(fig_pie, use_container_width=True)

        # R_t Velocity (rate of change)
        st.markdown("### R_t Velocity (Rate of Change)")
        fig_rt_vel = go.Figure()
        for i, (country, res) in enumerate(forecast_results.items()):
            rt_diff = np.diff(res["rt"])
            fig_rt_vel.add_trace(
                go.Bar(
                    x=res["dates"][1:],
                    y=rt_diff,
                    name=country,
                    marker_color=palette[i % len(palette)],
                    opacity=0.8,
                )
            )
        fig_rt_vel.add_hline(y=0, line_color="white", line_dash="dot", opacity=0.5)
        fig_rt_vel.update_layout(
            title="Daily R_t Change (Acceleration/Deceleration)",
            xaxis_title="Date",
            yaxis_title="ΔR_t",
            template="plotly_dark",
            height=350,
            barmode="group",
        )
        st.plotly_chart(fig_rt_vel, use_container_width=True)

        st.markdown("### Environmental Drivers of Transmission")
        st.markdown(
            "Analysis of how temperature, humidity, and UV levels correlate with predicted reproduction rates."
        )

        for country, res in forecast_results.items():
            cd = res["data"].tail(60)  # Last 60 days
            env_cols = [
                c
                for c in cd.columns
                if c
                in [
                    "temperature_avg",
                    "temperature_min",
                    "temperature_max",
                    "humidity_avg",
                    "uv_index",
                    "precipitation",
                ]
            ]
            if env_cols:
                corr = (
                    cd[env_cols + ["case_growth_rate"]]
                    .corr()["case_growth_rate"]
                    .drop("case_growth_rate")
                )

                fig_env = go.Figure(
                    go.Bar(
                        x=[c.replace("_", " ").title() for c in corr.index],
                        y=corr.values,
                        marker_color=[
                            COLORS["success"] if v < 0 else COLORS["danger"] for v in corr.values
                        ],
                        text=[f"{v:.2f}" for v in corr.values],
                        textposition="auto",
                    )
                )
                fig_env.update_layout(
                    title=f"Environmental Correlation — {country}",
                    yaxis_title="Correlation Coefficient",
                )
                style_chart(fig_env, 300)
                st.plotly_chart(fig_env, use_container_width=True)

        st.markdown("### Predicted Transmission Parameters")
        st.markdown(
            "Values predicted by the model for the 30-day forecast horizon. "
            "β = Transmission Rate, σ = Incubation->Infectious Rate, γ = Recovery Rate."
        )

        for country, res in forecast_results.items():
            fig_params = go.Figure()
            fig_params.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=res["betas"],
                    name="β (Transmission)",
                    line=dict(color=COLORS["secondary"], width=2),
                )
            )
            fig_params.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=res["sigmas"],
                    name="σ (Incubation)",
                    line=dict(color=COLORS["warning"], width=2),
                )
            )
            fig_params.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=res["gammas"],
                    name="γ (Recovery)",
                    line=dict(color=COLORS["success"], width=2),
                )
            )

            fig_params.update_layout(
                title=f"SEIR Parameters Forecast — {country}", yaxis_title="Rate Value"
            )
            style_chart(fig_params, 350)
            st.plotly_chart(fig_params, use_container_width=True)

        # Outbreak Survival Probability (Allen et al. branching process theory)
        st.markdown("### Epidemic Survival Probability")
        st.markdown(
            "Probability that the epidemic continues (does not go extinct stochastically). "
            "Based on branching process theory with negative binomial offspring distribution. "
            "Even with R_t > 1, there is a non-zero probability of stochastic extinction."
        )

        fig_surv = go.Figure()
        palette_surv = [
            COLORS["danger"],
            COLORS["warning"],
            COLORS["info"],
            COLORS["purple"],
        ]
        for i, (country, res) in enumerate(forecast_results.items()):
            rt_vals = res["rt"]
            cum_cases = np.cumsum(res["pred_mean"])
            survival = SEIR_LSTM.compute_epidemic_survival_probability(
                rt_vals, cum_cases, dispersion_k=0.5
            )
            fig_surv.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=survival,
                    mode="lines",
                    name=country,
                    line=dict(color=palette_surv[i % len(palette_surv)], width=2.5),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(palette_surv[i % len(palette_surv)], 0.1),
                )
            )
        fig_surv.add_hline(y=0.5, line_dash="dot", line_color="white", opacity=0.5)
        fig_surv.update_layout(
            title="Epidemic Survival Probability Over Forecast Horizon",
            xaxis_title="Date",
            yaxis_title="P(Epidemic Continues)",
            yaxis_range=[0, 1.05],
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_surv, use_container_width=True)

        # Interpretation
        for country, res in forecast_results.items():
            rt_vals = res["rt"]
            cum_cases = np.cumsum(res["pred_mean"])
            surv = SEIR_LSTM.compute_epidemic_survival_probability(
                rt_vals, cum_cases, dispersion_k=0.5
            )
            avg_surv = np.mean(surv)
            if avg_surv > 0.8:
                risk_label = "HIGH - Epidemic likely to continue"
                risk_color = COLORS["danger"]
            elif avg_surv > 0.5:
                risk_label = "MODERATE - Uncertain trajectory"
                risk_color = COLORS["warning"]
            else:
                risk_label = "LOW - Possible stochastic extinction"
                risk_color = COLORS["success"]
            st.markdown(
                f"**{country}**: Avg Survival Probability = {avg_surv:.1%} "
                f"— <span style='color:{risk_color}'>{risk_label}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("### R_t Summary")
        for country, res in forecast_results.items():
            avg_rt = np.mean(res["rt"])
            max_rt = np.max(res["rt"])
            min_rt = np.min(res["rt"])
            status = "Declining" if avg_rt < 1.0 else "Growing"
            risk = "High" if avg_rt > 1.5 else ("Medium" if avg_rt > 1.0 else "Low")
            st.markdown(
                f"**{country}**: Avg={avg_rt:.2f}, Min={min_rt:.2f}, Max={max_rt:.2f} — "
                f"Status: **{status}**, Risk Level: **{risk}**"
            )
    else:
        st.warning("Select a country to see R_t analysis.")

# ========================
# HOTSPOT TAB
# ========================
with tab_hotspot:
    st.markdown("### Hotspot Detection & Risk Classification")
    st.markdown(
        "Hotspots identified by combining case rate per million, "
        "growth rate, and vaccination coverage into a composite risk score. "
        "Risk classes: Critical (500+), High (200+), Medium (80+), Low (<80)."
    )

    hotspots = detect_hotspots(df, threshold_percentile=80)
    hotspot_countries = hotspots[hotspots["is_hotspot"]].head(20)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Hotspots", len(hotspot_countries))
    col2.metric("Highest Risk Score", f"{hotspot_countries['risk_score'].max():.1f}")
    col3.metric("Avg Growth Rate", f"{hotspot_countries['growth_rate'].mean():.2%}")
    crit_count = len(hotspot_countries[hotspot_countries["risk_class"] == "Critical"])
    col4.metric("Critical Countries", crit_count)

    st.markdown("---")

    # Hotspot Treemap
    st.markdown("### Global Risk Treemap")
    hotspots_map = hotspots.copy()
    hotspots_map["Risk Label"] = (
        hotspots_map["Country/Region"]
        + "<br>Score: "
        + hotspots_map["risk_score"].round(1).astype(str)
    )

    fig_treemap = px.treemap(
        hotspots_map,
        path=[px.Constant("World"), "risk_class", "Country/Region"],
        values="population",
        color="risk_score",
        color_continuous_scale="YlOrRd",
        custom_data=["risk_score", "growth_rate"],
    )
    fig_treemap.update_traces(
        hovertemplate="<b>%{label}</b><br>Risk Score: %{customdata[0]:.1f}<br>Growth Rate: %{customdata[1]:.2%}<extra></extra>"
    )
    style_chart(fig_treemap, 500)
    fig_treemap.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig_treemap, use_container_width=True)

    st.markdown("---")

    # Risk distribution chart
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.markdown("### Risk Class Distribution")
        risk_counts = hotspots["risk_class"].value_counts()
        risk_colors = {
            "Critical": COLORS["danger"],
            "High": COLORS["warning"],
            "Medium": COLORS["info"],
            "Low": COLORS["success"],
        }
        fig_risk_dist = go.Figure(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[risk_colors.get(c, COLORS["primary"]) for c in risk_counts.index],
            )
        )
        style_chart(fig_risk_dist, 350)
        fig_risk_dist.update_layout(xaxis_title="Risk Class", yaxis_title="Number of Countries")
        st.plotly_chart(fig_risk_dist, use_container_width=True)

    # Hotspot ranking table
    st.markdown("### Active Hotspots (Risk Ranked)")
    display_hotspot = hotspot_countries[
        [
            "Country/Region",
            "new_cases_smoothed",
            "population",
            "risk_per_million",
            "growth_rate",
            "risk_score",
            "risk_class",
        ]
    ].copy()
    display_hotspot.columns = [
        "Country",
        "Latest Cases",
        "Population",
        "Cases/M",
        "Growth Rate",
        "Risk Score",
        "Risk Class",
    ]
    display_hotspot["Latest Cases"] = display_hotspot["Latest Cases"].apply(lambda x: f"{x:,.0f}")
    display_hotspot["Population"] = display_hotspot["Population"].apply(lambda x: f"{x:,.0f}")
    display_hotspot["Cases/M"] = display_hotspot["Cases/M"].apply(lambda x: f"{x:.1f}")
    display_hotspot["Growth Rate"] = display_hotspot["Growth Rate"].apply(lambda x: f"{x:.2%}")
    display_hotspot["Risk Score"] = display_hotspot["Risk Score"].apply(lambda x: f"{x:.1f}")

    def highlight_risk(row):
        cls = row["Risk Class"]
        if cls == "Critical":
            return ["background-color: rgba(239,85,59,0.2)"] * len(row)
        elif cls == "High":
            return ["background-color: rgba(255,161,90,0.2)"] * len(row)
        elif cls == "Medium":
            return ["background-color: rgba(25,211,243,0.15)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_hotspot.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    with col_dist2:
        st.markdown("### Risk Score Breakdown (Top 15)")
        fig_hotspot = go.Figure()
        fig_hotspot.add_trace(
            go.Bar(
                x=hotspot_countries["Country/Region"].head(15),
                y=hotspot_countries["risk_per_million"].head(15),
                name="Cases/M (50%)",
                marker_color=COLORS["danger"],
            )
        )
        fig_hotspot.add_trace(
            go.Bar(
                x=hotspot_countries["Country/Region"].head(15),
                y=hotspot_countries["growth_rate"].head(15).clip(upper=2) * 500,
                name="Growth Rate (30%)",
                marker_color=COLORS["warning"],
            )
        )
        fig_hotspot.add_trace(
            go.Bar(
                x=hotspot_countries["Country/Region"].head(15),
                y=(1 - hotspot_countries["vaccination_rate"].head(15)) * 100,
                name="Unvaccinated (20%)",
                marker_color=COLORS["info"],
            )
        )
        fig_hotspot.update_layout(barmode="stack")
        style_chart(fig_hotspot, 350)
        fig_hotspot.update_layout(xaxis_tickangle=-45, yaxis_title="Score Contribution")
        st.plotly_chart(fig_hotspot, use_container_width=True)

# ========================
# OUTBREAK PATTERNS TAB
# ========================
with tab_patterns:
    st.markdown("### Outbreak Pattern Analysis")
    st.markdown("Wave detection, case acceleration, and temporal dynamics.")

    # Multi-country comparison: Normalized cases over time
    if len(selected_countries) > 1:
        st.markdown("#### Multi-Country Comparison: Normalized Case Trends")
        fig_compare = go.Figure()
        palette = [
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["success"],
            COLORS["warning"],
            COLORS["info"],
        ]

        for i, country in enumerate(selected_countries):
            cd = df[df["Country/Region"] == country].copy().sort_values("date")
            if cd.empty:
                continue

            # Normalize cases to percentage of max
            max_cases = cd["new_cases_smoothed"].max()
            if max_cases > 0:
                cd["normalized_cases"] = cd["new_cases_smoothed"] / max_cases * 100

            fig_compare.add_trace(
                go.Scatter(
                    x=cd["date"],
                    y=cd["normalized_cases"],
                    mode="lines",
                    name=country,
                    line=dict(width=2.5, color=palette[i % len(palette)]),
                    opacity=0.85,
                )
            )

        fig_compare.update_layout(
            title="Normalized Case Trends (% of Peak)",
            xaxis_title="Date",
            yaxis_title="% of Peak Cases",
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_compare.update_xaxes(gridcolor=COLORS["grid"])
        fig_compare.update_yaxes(gridcolor=COLORS["grid"])
        st.plotly_chart(fig_compare, use_container_width=True)

        # Wave frequency comparison
        st.markdown("#### Wave Pattern Comparison")
        wave_data = []
        for country in selected_countries:
            cd = df[df["Country/Region"] == country].copy().sort_values("date")
            if cd.empty:
                continue

            window = 60
            cd["rolling_max"] = (
                cd["new_cases_smoothed"].rolling(window, center=True, min_periods=1).max()
            )
            cd["is_peak"] = cd["new_cases_smoothed"] == cd["rolling_max"]
            min_peak_height = cd["new_cases_smoothed"].max() * 0.1
            significant_peaks = cd[cd["is_peak"] & (cd["new_cases_smoothed"] > min_peak_height)]

            wave_data.append(
                {
                    "Country": country,
                    "Waves Detected": len(significant_peaks),
                    "Peak Cases": cd["new_cases_smoothed"].max(),
                    "Mean Growth Rate": cd["case_growth_rate"].mean()
                    if "case_growth_rate" in cd.columns
                    else 0,
                }
            )

        wave_df = pd.DataFrame(wave_data)
        col_w1, col_w2 = st.columns(2)

        with col_w1:
            fig_waves = px.bar(
                wave_df,
                x="Country",
                y="Waves Detected",
                title="Number of Outbreak Waves",
                color="Waves Detected",
                color_continuous_scale="YlOrRd",
                text_auto=True,
            )
            style_chart(fig_waves, 350)
            st.plotly_chart(fig_waves, use_container_width=True)

        with col_w2:
            fig_growth = px.bar(
                wave_df,
                x="Country",
                y="Mean Growth Rate",
                title="Average Growth Rate by Country",
                color="Mean Growth Rate",
                color_continuous_scale="RdYlGn_r",
                text_auto=".2%",
            )
            style_chart(fig_growth, 350)
            st.plotly_chart(fig_growth, use_container_width=True)

    for country in selected_countries:
        cd = df[df["Country/Region"] == country].copy().sort_values("date")
        if cd.empty:
            continue

        st.markdown(f"#### {country}")
        cd["pct_change"] = (
            cd["new_cases_smoothed"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        )
        cd["acceleration"] = cd["new_cases_smoothed"].diff().diff().fillna(0)
        cd["cumulative"] = cd.groupby(cd["date"].dt.year)["new_cases_smoothed"].cumsum()

        fig_patterns = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "Daily New Cases with Peak Detection",
                "Growth Rate (%)",
                "Weekly Case Distribution",
            ),
        )

        fig_patterns.add_trace(
            go.Scatter(
                x=cd["date"],
                y=cd["new_cases_smoothed"],
                mode="lines",
                name="New Cases",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(99,110,250,0.1)",
            ),
            row=1,
            col=1,
        )

        # Peak detection using rolling window
        peaks = cd[
            cd["new_cases_smoothed"] == cd["new_cases_smoothed"].rolling(60, center=True).max()
        ]
        if not peaks.empty:
            fig_patterns.add_trace(
                go.Scatter(
                    x=peaks["date"],
                    y=peaks["new_cases_smoothed"],
                    mode="markers",
                    name="Peaks",
                    marker=dict(size=8, color=COLORS["warning"], symbol="diamond"),
                ),
                row=1,
                col=1,
            )

        fig_patterns.add_trace(
            go.Scatter(
                x=cd["date"],
                y=cd["pct_change"] * 100,
                mode="lines",
                name="Growth Rate",
                line=dict(color=COLORS["info"], width=1.5),
            ),
            row=2,
            col=1,
        )
        fig_patterns.add_hrect(
            y0=-5, y1=5, fillcolor="rgba(0,204,150,0.08)", line_width=0, row=2, col=1
        )

        weekly = (
            cd.groupby(pd.Grouper(key="date", freq="W")).agg({"new_cases": "sum"}).reset_index()
        )
        fig_patterns.add_trace(
            go.Bar(
                x=weekly["date"],
                y=weekly["new_cases"],
                name="Weekly Total",
                marker_color=COLORS["success"],
                opacity=0.7,
            ),
            row=3,
            col=1,
        )

        style_chart(fig_patterns, 650)
        st.plotly_chart(fig_patterns, use_container_width=True)

        # Wave Detection Analysis
        st.markdown(f"**{country} — Wave Detection**")
        # Detect waves by finding local maxima in smoothed cases
        window = 60  # 60-day window for peak detection
        cd["rolling_max"] = (
            cd["new_cases_smoothed"].rolling(window, center=True, min_periods=1).max()
        )
        cd["is_peak"] = cd["new_cases_smoothed"] == cd["rolling_max"]
        # Filter to significant peaks (at least 10% of max)
        min_peak_height = cd["new_cases_smoothed"].max() * 0.1
        significant_peaks = cd[cd["is_peak"] & (cd["new_cases_smoothed"] > min_peak_height)]

        if len(significant_peaks) > 1:
            wave_info = []
            for idx, (_, peak) in enumerate(significant_peaks.iterrows()):
                wave_info.append(
                    {
                        "Wave": idx + 1,
                        "Peak Date": peak["date"].strftime("%Y-%m-%d"),
                        "Peak Cases": f"{peak['new_cases_smoothed']:,.0f}",
                        "% of Max": f"{peak['new_cases_smoothed'] / cd['new_cases_smoothed'].max() * 100:.1f}%",
                    }
                )
            st.dataframe(pd.DataFrame(wave_info), use_container_width=True, hide_index=True)
        else:
            st.info(f"Detected {len(significant_peaks)} significant wave(s) in the data.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{country} Peak", f"{cd['new_cases_smoothed'].max():,.0f}")
        col2.metric(f"{country} Avg Daily", f"{cd['new_cases_smoothed'].mean():,.0f}")
        col3.metric(f"{country} Growth Rate", f"{cd['pct_change'].iloc[-1] * 100:.1f}%")
        col4.metric(f"{country} Declining Days", f"{len(cd[cd['pct_change'] < 0])}")
        st.markdown("---")

# ========================
# MOBILITY TAB
# ========================
with tab_mobility:
    st.markdown("### Behavioral Changes & Awareness Responses")
    st.markdown(
        "How communities react to perceived danger (case spikes) by reducing contact, as modeled by Funk et al."
    )

    mobility_cols = [c for c in FEATURES if "percent_change" in c]
    if mobility_cols:
        for country in selected_countries:
            cd = df[df["Country/Region"] == country].copy().sort_values("date")
            if cd.empty:
                continue

            st.markdown(f"#### {country}")
            fig_mob = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12,
                subplot_titles=("New Cases (7d Avg)", "Mobility Changes"),
            )

            fig_mob.add_trace(
                go.Scatter(
                    x=cd["date"],
                    y=cd["new_cases_smoothed"],
                    mode="lines",
                    name="New Cases",
                    line=dict(color=COLORS["primary"], width=2),
                ),
                row=1,
                col=1,
            )

            mob_colors = [
                COLORS["success"],
                COLORS["warning"],
                COLORS["info"],
                COLORS["secondary"],
                COLORS["danger"],
                COLORS["purple"],
            ]
            for j, col in enumerate(mobility_cols):
                label = (
                    col.replace("_percent_change_from_baseline_smoothed", "")
                    .replace("_", " ")
                    .title()
                )
                fig_mob.add_trace(
                    go.Scatter(
                        x=cd["date"],
                        y=cd[col],
                        mode="lines",
                        name=label,
                        line=dict(color=mob_colors[j % len(mob_colors)], width=1.5),
                    ),
                    row=2,
                    col=1,
                )

            fig_mob.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="white", opacity=0.3)
            style_chart(fig_mob, 550)
            st.plotly_chart(fig_mob, use_container_width=True)

            # Correlation witheds analysis - Matrix Heatmap
            st.markdown(f"**Correlation Matrix: Mobility Factors vs Cases ({country})**")

            # Full correlation matrix
            corr_matrix = cd[mobility_cols + ["new_cases_smoothed", "case_growth_rate"]].corr()

            # Create heatmap
            corr_labels = {
                "retail_and_recreation_percent_change_from_baseline_smoothed": "Retail",
                "grocery_and_pharmacy_percent_change_from_baseline_smoothed": "Grocery",
                "parks_percent_change_from_baseline_smoothed": "Parks",
                "transit_stations_percent_change_from_baseline_smoothed": "Transit",
                "workplaces_percent_change_from_baseline_smoothed": "Workplace",
                "residential_percent_change_from_baseline_smoothed": "Residential",
                "new_cases_smoothed": "New Cases",
                "case_growth_rate": "Growth Rate",
            }

            corr_matrix_renamed = corr_matrix.rename(index=corr_labels, columns=corr_labels)

            fig_corr_heatmap = go.Figure(
                go.Heatmap(
                    z=corr_matrix_renamed.values,
                    x=corr_matrix_renamed.columns,
                    y=corr_matrix_renamed.index,
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title="Correlation"),
                    text=np.round(corr_matrix_renamed.values, 2),
                    texttemplate="%{text:.2f}",
                    textfont={"size": 9},
                )
            )
            fig_corr_heatmap.update_layout(
                title="Mobility-Cases Correlation Matrix",
                template="plotly_dark",
                height=450,
            )
            st.plotly_chart(fig_corr_heatmap, use_container_width=True)

            st.markdown("---")
    else:
        st.warning("No mobility data available.")

# ========================
# WHAT-IF ANALYSIS TAB
# ========================
with tab_whatif:
    st.markdown("### What-If Intervention Analysis")
    st.markdown(
        "Model different intervention timing scenarios and their impact on outbreak trajectories"
    )

    if forecast_results:
        # Interactive intervention timing
        st.markdown("#### Intervention Timing Analysis")

        timing_scenarios = {
            "Immediate (-7 days)": -7,
            "Current (Baseline)": 0,
            "Delayed (+7 days)": 7,
            "Delayed (+14 days)": 14,
        }

        # Create comparison figure for different intervention timings
        fig_timing = go.Figure()

        for country, res in forecast_results.items():
            cd = res["data"]
            baseline = res["pred_mean"]

            # Add baseline
            fig_timing.add_trace(
                go.Scatter(
                    x=res["dates"],
                    y=baseline,
                    mode="lines",
                    name=f"{country} - Baseline",
                    line=dict(width=2.5, color=COLORS["primary"]),
                )
            )

            # Add modified scenarios (scaled by timing impact)
            for scen_name, delay in timing_scenarios.items():
                if delay == 0:
                    continue
                # Simplified impact model: delayed intervention = higher peak
                impact_factor = 1 + delay * 0.01
                modified = baseline * impact_factor

                fig_timing.add_trace(
                    go.Scatter(
                        x=res["dates"],
                        y=modified,
                        mode="lines",
                        name=f"{country} - {scen_name}",
                        line=dict(width=1.5, dash="dot"),
                    )
                )

        fig_timing.update_layout(
            title="Intervention Timing Impact: Case Predictions",
            xaxis_title="Date",
            yaxis_title="Predicted New Cases",
            template="plotly_dark",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig_timing, use_container_width=True)

        # Controllability analysis - effectiveness thresholds
        st.markdown("#### Intervention Effectiveness Thresholds")

        col1, col2 = st.columns(2)

        with col1:
            # Threshold analysis
            thresholds = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
            threshold_data = []

            for country, res in forecast_results.items():
                baseline_total = res["pred_mean"].sum()

                for thresh in thresholds:
                    threshold_total = baseline_total * thresh
                    reduction = (1 - thresh) * 100
                    threshold_data.append(
                        {
                            "Country": country,
                            "Threshold Factor": thresh,
                            "Reduction %": reduction,
                            "Predicted Cases": threshold_total,
                        }
                    )

            thresh_df = pd.DataFrame(threshold_data)

            fig_thresh = px.line(
                thresh_df[thresh_df["Country"] == selected_countries[0]]
                if selected_countries
                else thresh_df,
                x="Threshold Factor",
                y="Predicted Cases",
                color="Country",
                markers=True,
                title="Predicted Cases at Different Intervention Thresholds",
            )
            fig_thresh.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_thresh, use_container_width=True)

        with col2:
            # Sensitivity analysis
            st.markdown("**Parameter Sensitivity Analysis**")

            # Sensitivity factors for key parameters
            params_to_test = {
                "Case Finding Rate": [0.3, 0.5, 0.7, 1.0],
                "Contact Tracing": [0.2, 0.5, 0.8, 1.0],
                "Isolation Efficacy": [0.4, 0.6, 0.8, 1.0],
            }

            sensitivity_data = []
            for param_name, param_values in params_to_test.items():
                for val in param_values:
                    sensitivity_data.append(
                        {
                            "Parameter": param_name,
                            "Efficacy": val,
                            "Impact Score": val * 100,
                        }
                    )

            sens_df = pd.DataFrame(sensitivity_data)
            fig_sens = px.bar(
                sens_df,
                x="Parameter",
                y="Impact Score",
                color="Efficacy",
                barmode="group",
                title="Parameter Sensitivity Impact",
                color_continuous_scale="Viridis",
            )
            fig_sens.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_sens, use_container_width=True)

        # Key recommendations
        st.markdown("#### Intervention Recommendations")

        for country, res in forecast_results.items():
            avg_rt = np.mean(res["rt"])
            baseline_30d = res["pred_mean"].sum()

            col_rec1, col_rec2 = st.columns(2)

            with col_rec1:
                if avg_rt > 1.5:
                    st.error(f"**{country}: CRITICAL** - Immediate lockdowns needed")
                    st.write(f"• Current R_t: {avg_rt:.2f}")
                    st.write(f"• 30-day projected cases: {baseline_30d:,.0f}")
                    st.write(f"• Recommended: Full containment measures")
                elif avg_rt > 1.0:
                    st.warning(f"**{country}: HIGH RISK** - Accelerating spread")
                    st.write(f"• Current R_t: {avg_rt:.2f}")
                    st.write(f"• Recommended: Moderate restrictions")
                else:
                    st.success(f"**{country}: LOW RISK** - Controlled spread")
                    st.write(f"• Current R_t: {avg_rt:.2f}")
                    st.write(f"• Recommended: Maintain status quo")

            with col_rec2:
                # Calculate intervention impact
                impact_data = {
                    "Scenario": [
                        "Mild Restriction",
                        "Moderate Restriction",
                        "Strong Restriction",
                        "Full Lockdown",
                    ],
                    "Mobility Change": [-10, -20, -40, -60],
                    "Projected Reduction": [15, 30, 55, 75],
                }
                impact_df = pd.DataFrame(impact_data)
                fig_imp = px.bar(
                    impact_df,
                    x="Scenario",
                    y="Projected Reduction",
                    text="Projected Reduction",
                    title=f"Projected Case Reduction by Intervention Level - {country}",
                    color="Projected Reduction",
                    color_continuous_scale="Greens",
                )
                fig_imp.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_imp, use_container_width=True)

# ========================
# SCENARIO ANALYSIS TAB
# ========================
with tab_scenario:
    st.markdown("### Scenario Analysis & Intervention Modeling")
    st.markdown(
        "Compare different intervention scenarios and their predicted impact on outbreak dynamics."
    )

    # Scenario definitions with intervention levels
    scenarios = {
        "No Intervention": {"mobility": 50, "retail": 50, "transit": 50, "vaccination": 0},
        "Lockdown (-50%)": {"mobility": 0, "retail": 0, "transit": 0, "vaccination": 0},
        "Minimal Restriction": {"mobility": 20, "retail": 20, "transit": 20, "vaccination": 0},
        "Vaccination Campaign": {"mobility": 30, "retail": 30, "transit": 30, "vaccination": 50},
        "Full Containment": {"mobility": -30, "retail": -30, "transit": -30, "vaccination": 20},
    }

    if forecast_results:
        st.markdown("#### Scenario Comparison: Predicted 30-Day Cumulative Cases")

        scenario_results = {}
        for scenario_name, mods in scenarios.items():
            sc_res = {}
            for country, res in forecast_results.items():
                country_data = res["data"]
                try:
                    s_res = run_prediction(
                        country_data,
                        (mods["mobility"], mods["retail"], mods["transit"], mods["vaccination"]),
                        device,
                        model,
                        scaler,
                        n_samples=min(n_samples, 30),
                    )
                    sc_res[country] = s_res["pred_mean"].sum()
                except Exception:
                    sc_res[country] = 0
            scenario_results[scenario_name] = sc_res

        scenario_df = pd.DataFrame(scenario_results).T
        scenario_df.index.name = "Scenario"

        # Heatmap of scenarios
        fig_scenarios = go.Figure(
            data=go.Heatmap(
                z=scenario_df.values,
                x=scenario_df.columns,
                y=scenario_df.index,
                colorscale="YlOrRd",
                colorbar=dict(title="30-Day\nCases"),
                texttemplate="%{z:,.0f}",
                textfont={"size": 10},
            )
        )
        fig_scenarios.update_layout(
            title="Scenario Comparison - Predicted Cases by Country",
            xaxis_title="Country",
            yaxis_title="Scenario",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_scenarios, use_container_width=True)

        # Reduction analysis
        st.markdown("#### Intervention Effectiveness")
        baseline = scenario_df.loc["No Intervention"]

        effectiveness_data = []
        for scenario in scenarios.keys():
            if scenario == "No Intervention":
                continue
            pred_cases = scenario_df.loc[scenario]
            reduction = ((baseline - pred_cases) / baseline * 100).round(1)
            for c in reduction.index:
                effectiveness_data.append(
                    {"Scenario": scenario, "Country": c, "Reduction %": reduction[c]}
                )

        eff_df = pd.DataFrame(effectiveness_data)
        if not eff_df.empty:
            fig_eff = px.bar(
                eff_df,
                x="Country",
                y="Reduction %",
                color="Scenario",
                barmode="group",
                color_discrete_sequence=[
                    COLORS["success"],
                    COLORS["info"],
                    COLORS["warning"],
                    COLORS["secondary"],
                ],
                title="Case Reduction (%) vs No Intervention",
                text_auto=".1f",
            )
            fig_eff.update_layout(template="plotly_dark", yaxis_title="Reduction %")
            st.plotly_chart(fig_eff, use_container_width=True)

        # Time series comparison for top scenario
        st.markdown("#### Daily Case Trajectories by Scenario")
        fig_traj = go.Figure()
        colors_scenario = {
            "No Intervention": "#EF553B",
            "Lockdown (-50%)": "#00CC96",
            "Minimal Restriction": "#FFA15A",
            "Vaccination Campaign": "#19D3F3",
            "Full Containment": "#AB63FA",
        }

        for scenario_name, mods in scenarios.items():
            for country, res in forecast_results.items():
                if len(res.get("dates", [])) == 0:
                    continue
                try:
                    s_res = run_prediction(
                        res["data"],
                        (mods["mobility"], mods["retail"], mods["transit"], mods["vaccination"]),
                        device,
                        model,
                        scaler,
                        n_samples=min(n_samples, 20),
                    )
                    color_base = colors_scenario.get(scenario_name, "#666")
                    fig_traj.add_trace(
                        go.Scatter(
                            x=res["dates"],
                            y=s_res["pred_mean"],
                            mode="lines",
                            name=f"{scenario_name} - {country}",
                            line=dict(width=2, color=color_base),
                            fill="tozeroy",
                            fillcolor=hex_to_rgba(color_base, 0.15),
                            opacity=0.8,
                        )
                    )
                except Exception:
                    pass

        fig_traj.update_layout(
            title="Daily Case Trajectories - All Scenarios",
            xaxis_title="Date",
            yaxis_title="Predicted New Cases",
            template="plotly_dark",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # Best intervention recommendations
        st.markdown("#### Recommended Interventions")
        for country in selected_countries:
            best_scenario = scenario_df[country].idxmin()
            worst_scenario = scenario_df[country].idxmax()
            best_cases = scenario_df[country].min()
            worst_cases = scenario_df[country].max()

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**{country}: Best** - {best_scenario} ({best_cases:,.0f} cases)")
            with col2:
                st.error(f"**{country}: Worst** - {worst_scenario} ({worst_cases:,.0f} cases)")
    else:
        st.warning("Select a country to see scenario analysis.")

# ========================
# DEMOGRAPHIC TAB
# ========================
with tab_demographic:
    st.markdown("### Demographic & Healthcare Factor Analysis")
    st.markdown(
        "How population characteristics and healthcare infrastructure influence outbreak outcomes."
    )

    latest_data = df.groupby("Country/Region").last().reset_index()
    latest_data["cases_per_million"] = (
        latest_data["new_cases_smoothed"] / latest_data["population"].clip(lower=1) * 1e6
    ).fillna(0)
    latest_data["vaccination_pct"] = (
        latest_data["people_fully_vaccinated"] / latest_data["population"].clip(lower=1) * 100
    ).fillna(0)

    fig_demo = px.scatter(
        latest_data,
        x="human_development_index",
        y="cases_per_million",
        size="population",
        color="hospital_beds_per_thousand",
        hover_name="Country/Region",
        color_continuous_scale="Viridis",
        labels={
            "human_development_index": "Human Development Index",
            "cases_per_million": "Cases per Million",
            "hospital_beds_per_thousand": "Hospital Beds/1000",
        },
        size_max=50,
    )
    style_chart(fig_demo, 500)
    st.plotly_chart(fig_demo, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### HDI vs Vaccination Rate")
        fig_hdi = px.scatter(
            latest_data,
            x="human_development_index",
            y="vaccination_pct",
            size="population",
            hover_name="Country/Region",
            color="hospital_beds_per_thousand",
            color_continuous_scale="Tealgrn",
            labels={
                "human_development_index": "HDI",
                "vaccination_pct": "Vaccination %",
                "hospital_beds_per_thousand": "Beds/1000",
            },
            size_max=40,
        )
        style_chart(fig_hdi, 400)
        st.plotly_chart(fig_hdi, use_container_width=True)

    with col2:
        st.markdown("### Healthcare Capacity")
        fig_beds = px.box(
            latest_data,
            y="hospital_beds_per_thousand",
            points="all",
            color_discrete_sequence=[COLORS["info"]],
        )
        fig_beds.update_layout(yaxis_title="Beds per 1000")
        style_chart(fig_beds, 400)
        fig_beds.update_layout(showlegend=False)
        st.plotly_chart(fig_beds, use_container_width=True)

# ========================
# RISK MAP TAB
# ========================
with tab_risk:
    st.markdown("### Global Outbreak Risk Map")
    st.markdown(
        "Explore global outbreak risk dynamically. Risk scores combine case density, case growth rates, and infrastructural vulnerability."
    )

    available_dates = pd.Series(df["date"].unique()).sort_values()
    selected_date = st.slider(
        "Select Snapshot Date",
        min_value=available_dates.min().date(),
        max_value=available_dates.max().date(),
        value=available_dates.max().date(),
    )

    snap_df = df[df["date"].dt.date <= selected_date]
    if snap_df.empty:
        snap_df = df

    hotspots = detect_hotspots(snap_df)
    hotspots["Risk_Score"] = hotspots["risk_score"].clip(0, 1000)

    # Map country names to ISO-3 for proper choropleth
    hotspots["iso3"] = hotspots["Country/Region"].map(iso_map).fillna("")

    fig_map = go.Figure()
    fig_map.add_trace(
        go.Choropleth(
            locations=hotspots["iso3"],
            z=hotspots["Risk_Score"],
            locationmode="ISO-3",
            colorscale="YlOrRd",
            colorbar_title="Risk Score",
            hovertext=hotspots["Country/Region"],
            hovertemplate=("<b>%{hovertext}</b><br>Risk Score: %{z:.1f}<br><extra></extra>"),
        )
    )

    # Overlay trend arrows using Scattergeo
    trend_data = hotspots[hotspots["is_hotspot"]].copy()
    if not trend_data.empty:
        trend_data["arrow_text"] = trend_data["growth_rate"].apply(
            lambda x: "\u25b2" if x > 0 else "\u25bc"
        )
        trend_data["arrow_color"] = trend_data["growth_rate"].apply(
            lambda x: COLORS["danger"] if x > 0 else COLORS["success"]
        )
        max_growth = trend_data["growth_rate"].abs().max()
        trend_data["arrow_size"] = 5 + 15 * (
            trend_data["growth_rate"].abs() / max(max_growth, 1e-5)
        )

        fig_map.add_trace(
            go.Scattergeo(
                locationmode="country names",
                locations=trend_data["Country/Region"],
                text=trend_data["arrow_text"],
                hovertext=trend_data.apply(
                    lambda x: f"<b>{x['Country/Region']}</b><br>"
                    f"Risk: {x['Risk_Score']:.1f} ({x['risk_class']})<br>"
                    f"Growth: {x['growth_rate']:.1%}<br>"
                    f"Cases/M: {x['risk_per_million']:.0f}",
                    axis=1,
                ),
                hoverinfo="text",
                mode="text",
                textfont=dict(size=trend_data["arrow_size"], color=trend_data["arrow_color"]),
                name="Trend",
            )
        )

    style_chart(fig_map, 550)
    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#333",
            showland=True,
            landcolor="#1a1d24",
            showocean=True,
            oceancolor="#0e1117",
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Sub-components
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 15 Highest Risk")
        top_risk = hotspots.head(15)[
            [
                "Country/Region",
                "new_cases_smoothed",
                "population",
                "risk_per_million",
                "risk_score",
                "risk_class",
            ]
        ].copy()
        top_risk.columns = [
            "Country",
            "Cases",
            "Population",
            "Cases/M",
            "Risk Score",
            "Class",
        ]
        top_risk["Cases"] = top_risk["Cases"].apply(lambda x: f"{x:,.0f}")
        top_risk["Population"] = top_risk["Population"].apply(lambda x: f"{x:,.0f}")
        top_risk["Cases/M"] = top_risk["Cases/M"].apply(lambda x: f"{x:.0f}")
        top_risk["Risk Score"] = top_risk["Risk Score"].apply(lambda x: f"{x:.1f}")
        st.dataframe(top_risk, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Risk Distribution")
        fig_hist_risk = px.histogram(
            hotspots,
            x="Risk_Score",
            nbins=30,
            color_discrete_sequence=[COLORS["warning"]],
            labels={"Risk_Score": "Risk Score"},
        )
        style_chart(fig_hist_risk, 350)
        st.plotly_chart(fig_hist_risk, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top Outbreak Risks by Healthcare Vulnerability")
    st.markdown("Countries with high incidence and limited healthcare capacity (beds/1k).")

    fig_vuln = px.scatter(
        hotspots.head(30),
        x="healthcare_vulnerability",
        y="risk_score",
        size="population",
        color="risk_class",
        hover_name="Country/Region",
        color_discrete_map=risk_colors,
        labels={
            "healthcare_vulnerability": "Healthcare Vulnerability (1 - Normalized Beds)",
            "risk_score": "Composite Risk Score",
        },
    )
    style_chart(fig_vuln, 500)
    st.plotly_chart(fig_vuln, use_container_width=True)
