import pandas as pd
import numpy as np
import torch
import joblib
import os
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import find_peaks
import logging

from config import PATHS, FEATURES, TRAIN_CONFIG, get_device
from models.seir_lstm import SEIR_LSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Publication-quality style with dark theme
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#1a1d24",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.5,
        "font.size": 10,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.titleweight": "bold",
    }
)

# Color palette
COLORS = {
    "primary": "#636EFA",
    "secondary": "#EF553B",
    "success": "#00CC96",
    "warning": "#FFA15A",
    "danger": "#EF553B",
    "info": "#19D3F3",
    "purple": "#AB63FA",
    "S": "#636EFA",
    "E": "#FFA15A",
    "I": "#EF553B",
    "R": "#00CC96",
}


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-5)) * 100


def peak_timing_accuracy(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("inf")
    return abs(np.argmax(y_true) - np.argmax(y_pred))


def peak_magnitude_accuracy(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("inf")
    true_peak = np.max(y_true)
    if true_peak == 0:
        return abs(np.max(y_pred))
    return abs(true_peak - np.max(y_pred)) / true_peak


def directional_accuracy(y_true, y_pred):
    """Fraction of timesteps where direction of change matches."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    return np.mean(true_dir == pred_dir)


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error - more robust to outliers."""
    denominator = np.abs(y_true)
    weights = denominator / (denominator.sum() + 1e-10)
    return np.sum(weights * np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100


def forecast_horizon_accuracy(y_true, y_pred, horizons=[7, 14, 30]):
    """Calculate accuracy at different forecast horizons."""
    results = {}
    actual_cases = np.array(y_true)
    pred_cases = np.array(y_pred)

    for h in horizons:
        if h <= len(actual_cases):
            mae_h = np.mean(np.abs(actual_cases[:h] - pred_cases[:h]))
            results[f"MAE@{h}d"] = mae_h
            if h > 1:
                actual_diff = np.diff(actual_cases[:h])
                pred_diff = np.diff(pred_cases[:h])
                dir_acc = np.mean((actual_diff > 0) == (pred_diff > 0))
                results[f"DirAcc@{h}d"] = dir_acc
    return results


def evaluate():
    device = get_device()
    logger.info(f"Evaluating on {device}...")

    if not os.path.exists(PATHS.merged_data):
        logger.error("Merged data not found. Run preprocessing first.")
        return

    df = pd.read_csv(PATHS.merged_data)
    df["date"] = pd.to_datetime(df["date"])

    val_df = df[df["date"] >= TRAIN_CONFIG.train_split_date].copy()
    if val_df.empty:
        logger.error("No validation data found. Check train_split_date.")
        return

    if not os.path.exists(PATHS.scaler):
        logger.error(f"Scaler not found at {PATHS.scaler}. Run training first.")
        return
    scaler = joblib.load(PATHS.scaler)
    try:
        model, model_config = SEIR_LSTM.load_with_metadata(PATHS.model_weights, device)
    except FileNotFoundError:
        logger.error(f"Model weights not found at {PATHS.model_weights}. Run training first.")
        return
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    available_countries = sorted(val_df["Country/Region"].unique())
    benchmark_countries = [
        c
        for c in ["US", "United Kingdom", "Brazil", "India", "Germany", "Japan"]
        if c in available_countries
    ]
    if len(benchmark_countries) < 2:
        benchmark_countries = available_countries[:6]

    results = []
    seq_len = TRAIN_CONFIG.sequence_length

    for country in benchmark_countries:
        logger.info(f"Evaluating {country}...")
        country_data = (
            df[df["Country/Region"] == country].sort_values("date").reset_index(drop=True)
        )
        val_split_mask = country_data["date"] >= TRAIN_CONFIG.train_split_date
        if not val_split_mask.any():
            continue
        val_idx = val_split_mask.idxmax()

        start_idx = val_idx - seq_len
        if start_idx < 0:
            continue

        available_after = len(country_data) - val_idx
        forecast_len = min(seq_len, available_after)
        if forecast_len < 1:
            continue

        end_idx = val_idx + forecast_len
        test_window = country_data.iloc[start_idx:end_idx]

        if len(test_window) < seq_len + 1:
            continue

        pop = test_window["population"].iloc[0]
        actual_cases = test_window["new_cases_smoothed"].values[seq_len : seq_len + forecast_len]
        dates_forecast = test_window["date"].values[seq_len : seq_len + forecast_len]

        feats = scaler.transform(test_window[FEATURES])
        X_future = torch.tensor(feats[:seq_len], dtype=torch.float32).unsqueeze(0).to(device)

        vacc_values = test_window["new_vaccinations_smoothed"].values[
            seq_len : seq_len + forecast_len
        ]
        V_future = (
            torch.tensor(vacc_values / max(pop, 1), dtype=torch.float32).unsqueeze(0).to(device)
        )

        I_0 = test_window["new_cases_smoothed"].iloc[seq_len - 1]
        E_0 = I_0 * TRAIN_CONFIG.e_multiplier
        R_0 = test_window["total_cases"].iloc[seq_len - 1]
        S_0 = max(pop - E_0 - I_0 - R_0, 0)

        with torch.no_grad():
            preds, params = model(
                X_future,
                (
                    torch.tensor([S_0], device=device),
                    torch.tensor([E_0], device=device),
                    torch.tensor([I_0], device=device),
                    torch.tensor([R_0], device=device),
                ),
                torch.tensor([pop], device=device),
                V_future,
            )
            pred_cases = preds[0].cpu().numpy()
            betas = params[0][0].cpu().numpy()
            sigmas = params[1][0].cpu().numpy()
            gammas = params[2][0].cpu().numpy()
            rt_pred = betas / np.maximum(gammas, 1e-5)

        actual_cases = actual_cases[: len(pred_cases)]
        dates_forecast = dates_forecast[: len(pred_cases)]

        mae = mean_absolute_error(actual_cases, pred_cases)
        rmse = np.sqrt(mean_squared_error(actual_cases, pred_cases))
        mape_val = mape(actual_cases, pred_cases)
        smape_val = smape(actual_cases, pred_cases)
        r2 = 1 - np.sum((actual_cases - pred_cases) ** 2) / (
            np.sum((actual_cases - np.mean(actual_cases)) ** 2) + 1e-10
        )
        peak_timing_err = peak_timing_accuracy(actual_cases, pred_cases)
        peak_magnitude_err = peak_magnitude_accuracy(actual_cases, pred_cases)
        dir_acc = directional_accuracy(actual_cases, pred_cases)
        wmape_val = wmape(actual_cases, pred_cases)
        horizon_acc = forecast_horizon_accuracy(actual_cases, pred_cases)

        result_entry = {
            "Country": country,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape_val,
            "SMAPE": smape_val,
            "WMAPE": wmape_val,
            "R2": r2,
            "PeakTimingErr": peak_timing_err,
            "PeakMagErr": peak_magnitude_err,
            "DirAccuracy": dir_acc,
        }
        # Add horizon-specific metrics
        for key, val in horizon_acc.items():
            result_entry[key] = val
        results.append(result_entry)

        # ============================================================
        # COMPREHENSIVE DIAGNOSTIC FIGURE (3x3 grid)
        # ============================================================
        fig = plt.figure(figsize=(22, 18))
        fig.suptitle(
            f"Epidemic Forecast Diagnostic — {country}", fontsize=18, fontweight="bold", y=0.99
        )
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1.0, 1.0])

        # --- 1. Forecast vs Actual with Uncertainty Fan Chart (large, spans 2 columns) ---
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            dates_forecast,
            actual_cases,
            "o-",
            label="Actual",
            color=COLORS["success"],
            markersize=5,
            linewidth=2.5,
            zorder=4,
        )
        ax1.plot(
            dates_forecast,
            pred_cases,
            "s--",
            label="Predicted Mean",
            color=COLORS["secondary"],
            markersize=4,
            linewidth=2.2,
            zorder=4,
        )

        # Multi-level uncertainty fan chart (50%, 80%, 95% CIs)
        residual_std = np.std(actual_cases - pred_cases)
        ci_levels = [
            (1.96, 0.08, "95% CI"),
            (1.28, 0.12, "80% CI"),
            (0.67, 0.18, "50% CI"),
        ]
        for z_score, alpha, label in ci_levels:
            upper = pred_cases + z_score * residual_std
            lower = np.maximum(pred_cases - z_score * residual_std, 0)
            ax1.fill_between(
                dates_forecast,
                lower,
                upper,
                alpha=alpha,
                color=COLORS["secondary"],
                label=label,
                zorder=1,
            )

        # Mark peaks
        peak_idx_true = np.argmax(actual_cases)
        peak_idx_pred = np.argmax(pred_cases)
        ax1.axvline(
            dates_forecast[peak_idx_true],
            color=COLORS["success"],
            alpha=0.3,
            linestyle=":",
            linewidth=1,
        )
        ax1.axvline(
            dates_forecast[peak_idx_pred],
            color=COLORS["secondary"],
            alpha=0.3,
            linestyle=":",
            linewidth=1,
        )
        ax1.annotate(
            f"Actual Peak: {actual_cases[peak_idx_true]:,.0f}",
            xy=(dates_forecast[peak_idx_true], actual_cases[peak_idx_true]),
            xytext=(10, 15),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["success"],
            arrowprops=dict(arrowstyle="->", color=COLORS["success"]),
        )
        ax1.annotate(
            f"Pred Peak: {pred_cases[peak_idx_pred]:,.0f}",
            xy=(dates_forecast[peak_idx_pred], pred_cases[peak_idx_pred]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["secondary"],
            arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
        )
        ax1.set_title("Forecast vs Actual (30-Day Horizon) with Confidence Fan", fontweight="bold")
        ax1.legend(framealpha=0.3, loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.tick_params(axis="x", rotation=30)
        ax1.set_ylabel("New Cases (7d avg)")

        # --- 2. Metrics summary (text panel) ---
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        wmape_val = wmape(actual_cases, pred_cases)
        horizon_acc = forecast_horizon_accuracy(actual_cases, pred_cases)
        metrics_text = (
            f"MAE:       {mae:,.0f}\n"
            f"RMSE:      {rmse:,.0f}\n"
            f"MAPE:      {mape_val:.1f}%\n"
            f"SMAPE:     {smape_val:.1f}%\n"
            f"WMAPE:     {wmape_val:.1f}%\n"
            f"R²:        {r2:.4f}\n"
            f"Peak Err:  {peak_timing_err} days\n"
            f"Peak Mag:  {peak_magnitude_err:.2%}\n"
            f"Dir Acc:   {dir_acc:.1%}"
        )
        # Add horizon-specific accuracy
        for key, val in horizon_acc.items():
            if "MAE" in key:
                metrics_text += f"\n{key}: {val:,.0f}"
            else:
                metrics_text += f"\n{key}: {val:.1%}"

        bbox_props = dict(
            boxstyle="round,pad=0.8", facecolor="#1a1d24", edgecolor="#555", linewidth=1.5
        )
        ax2.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=bbox_props,
        )
        ax2.set_title("Metrics Summary", fontweight="bold")

        # Model info
        model_info = (
            f"Model: Attention-SEIR-LSTM\n"
            f"Hidden: {model_config.get('hidden_dim', 64)}\n"
            f"Layers: {model_config.get('num_layers', 2)}\n"
            f"Seq Len: {seq_len}"
        )
        ax2.text(
            0.05,
            0.18,
            model_info,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            color="#888",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#15171d", edgecolor="#444"),
        )

        # --- 3. R_t over time with survival probability ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(dates_forecast, rt_pred, "-", color=COLORS["info"], linewidth=2.5, label="R_t")
        ax3.axhline(y=1.0, color=COLORS["secondary"], linestyle="--", alpha=0.7, linewidth=1.5)
        # Shade growing/declining zones
        ax3.fill_between(
            dates_forecast, 0, 1, alpha=0.08, color=COLORS["success"], label="Declining (R_t<1)"
        )
        ax3.fill_between(
            dates_forecast,
            1,
            max(rt_pred.max() * 1.1, 2),
            alpha=0.08,
            color=COLORS["secondary"],
            label="Growing (R_t>1)",
        )
        ax3.set_title("Effective Reproduction Number (R_t)", fontweight="bold")
        ax3.set_ylabel("R_t")
        ax3.legend(framealpha=0.3, fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax3.tick_params(axis="x", rotation=30)
        ax3.set_ylim(0, max(rt_pred.max() * 1.2, 2.5))

        # --- 4. SEIR Parameters ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(
            dates_forecast,
            betas,
            "-",
            label="β (Transmission)",
            color=COLORS["secondary"],
            linewidth=2,
        )
        ax4.plot(
            dates_forecast,
            sigmas,
            "-",
            label="σ (Incubation)",
            color=COLORS["warning"],
            linewidth=2,
        )
        ax4.plot(
            dates_forecast, gammas, "-", label="γ (Recovery)", color=COLORS["success"], linewidth=2
        )
        ax4.set_title("SEIR Parameters", fontweight="bold")
        ax4.legend(framealpha=0.3, fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax4.tick_params(axis="x", rotation=30)
        ax4.set_ylabel("Rate")

        # --- 5. Outbreak Survival Probability (from PGF branching process) ---
        ax5 = fig.add_subplot(gs[1, 2])
        cumulative_forecast = np.cumsum(pred_cases)
        survival_probs = SEIR_LSTM.compute_epidemic_survival_probability(
            rt_pred, cumulative_forecast, dispersion_k=0.5
        )
        ax5.fill_between(
            dates_forecast,
            0,
            survival_probs,
            alpha=0.3,
            color=COLORS["danger"],
        )
        ax5.plot(
            dates_forecast,
            survival_probs,
            "-",
            color=COLORS["danger"],
            linewidth=2.5,
            label="P(Survival)",
        )
        ax5.axhline(y=0.5, color="white", linestyle=":", alpha=0.5, linewidth=1)
        ax5.set_title("Epidemic Survival Probability", fontweight="bold")
        ax5.set_ylabel("P(Epidemic Continues)")
        ax5.set_ylim(0, 1.05)
        ax5.legend(framealpha=0.3, fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax5.tick_params(axis="x", rotation=30)
        ax5.text(
            0.98,
            0.02,
            "Based on branching process\ntheory (Allen et al.)",
            transform=ax5.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="#888",
            style="italic",
        )

        # --- 6. SEIR Compartment simulation (full width) ---
        ax6 = fig.add_subplot(gs[2, :])
        with torch.no_grad():
            seir_data = model.get_seir_compartments(
                X_future,
                (
                    torch.tensor([S_0], device=device),
                    torch.tensor([E_0], device=device),
                    torch.tensor([I_0], device=device),
                    torch.tensor([R_0], device=device),
                ),
                torch.tensor([pop], device=device),
                V_future,
            )
        S_traj = seir_data["S"][0].cpu().numpy()
        E_traj = seir_data["E"][0].cpu().numpy()
        I_traj = seir_data["I"][0].cpu().numpy()
        R_traj = seir_data["R"][0].cpu().numpy()

        ax6.plot(
            dates_forecast,
            S_traj / pop * 100,
            "-",
            label="S (Susceptible)",
            color=COLORS["S"],
            linewidth=2,
        )
        ax6.plot(
            dates_forecast,
            E_traj / pop * 100,
            "-",
            label="E (Exposed)",
            color=COLORS["E"],
            linewidth=2,
        )
        ax6.plot(
            dates_forecast,
            I_traj / pop * 100,
            "-",
            label="I (Infectious)",
            color=COLORS["I"],
            linewidth=2,
        )
        ax6.plot(
            dates_forecast,
            R_traj / pop * 100,
            "-",
            label="R (Recovered)",
            color=COLORS["R"],
            linewidth=2,
        )
        ax6.set_title("SEIR Compartment Dynamics (% of Population)", fontweight="bold")
        ax6.set_ylabel("Population %")
        ax6.legend(framealpha=0.3, ncol=4, fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax6.tick_params(axis="x", rotation=30)
        ax6.set_xlabel("Date")

        # Population conservation check
        total_pop_check = S_traj + E_traj + I_traj + R_traj
        conservation_err = np.abs(total_pop_check - pop).max() / pop * 100
        ax6.text(
            0.98,
            0.02,
            f"Conservation error: {conservation_err:.4f}%",
            transform=ax6.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            color="#888",
            bbox=dict(facecolor="#15171d", edgecolor="#444", boxstyle="round,pad=0.3"),
        )

        os.makedirs(PATHS.evaluation_plots, exist_ok=True)
        fig.savefig(
            f"{PATHS.evaluation_plots}/{country}_eval.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

        logger.info(
            f"  {country}: MAE={mae:.0f} | RMSE={rmse:.0f} | R²={r2:.4f} | DirAcc={dir_acc:.1%}"
        )

    # ============================================================
    # SUMMARY VISUALIZATIONS
    # ============================================================
    results_df = pd.DataFrame(results)

    # Cross-country comparison figure
    if len(results_df) > 1:
        fig_comp, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig_comp.suptitle("Cross-Country Evaluation Summary", fontsize=16, fontweight="bold")

        # MAE comparison
        ax = axes[0, 0]
        bars = ax.bar(results_df["Country"], results_df["MAE"], color=COLORS["primary"], alpha=0.8)
        ax.set_title("MAE by Country")
        ax.set_ylabel("Mean Absolute Error")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, results_df["MAE"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#e0e0e0",
            )

        # R² comparison
        ax = axes[0, 1]
        r2_colors = [
            COLORS["success"] if r > 0.5 else COLORS["warning"] if r > 0 else COLORS["secondary"]
            for r in results_df["R2"]
        ]
        bars = ax.bar(results_df["Country"], results_df["R2"], color=r2_colors, alpha=0.8)
        ax.set_title("R² by Country")
        ax.set_ylabel("R² Score")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)

        # Direction Accuracy
        ax = axes[0, 2]
        bars = ax.bar(
            results_df["Country"], results_df["DirAccuracy"] * 100, color=COLORS["info"], alpha=0.8
        )
        ax.set_title("Directional Accuracy by Country")
        ax.set_ylabel("Accuracy (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, 100)

        # MAPE comparison
        ax = axes[1, 0]
        bars = ax.bar(results_df["Country"], results_df["MAPE"], color=COLORS["warning"], alpha=0.8)
        ax.set_title("MAPE by Country")
        ax.set_ylabel("MAPE (%)")
        ax.tick_params(axis="x", rotation=30)

        # WMAPE comparison
        ax = axes[1, 1]
        if "WMAPE" in results_df.columns:
            bars = ax.bar(
                results_df["Country"], results_df["WMAPE"], color=COLORS["purple"], alpha=0.8
            )
            ax.set_title("WMAPE by Country")
            ax.set_ylabel("WMAPE (%)")
            ax.tick_params(axis="x", rotation=30)
        else:
            # Radar-style metrics comparison
            metrics_for_radar = ["MAE", "RMSE", "R2", "DirAccuracy"]
            x_pos = np.arange(len(results_df))
            width = 0.2
            for i, metric in enumerate(metrics_for_radar):
                vals = results_df[metric].values
                if vals.max() > 0:
                    vals_norm = vals / vals.max()
                else:
                    vals_norm = vals
                ax.bar(x_pos + i * width, vals_norm, width, label=metric, alpha=0.7)
            ax.set_xticks(x_pos + width * 1.5)
            ax.set_xticklabels(results_df["Country"], rotation=30)
            ax.legend(fontsize=7)
            ax.set_title("Normalized Metrics Comparison")

        # R_t summary heatmap
        ax = axes[1, 2]
        rt_data = []
        countries_eval = []
        for country in benchmark_countries:
            country_res = [r for r in results if r["Country"] == country]
            if country_res:
                countries_eval.append(country)
                rt_data.append(
                    [
                        country_res[0].get("MAE", 0),
                        country_res[0].get("RMSE", 0),
                        country_res[0].get("R2", 0),
                        country_res[0].get("DirAccuracy", 0) * 100,
                    ]
                )
        if rt_data:
            rt_arr = np.array(rt_data)
            # Normalize each metric column
            for col in range(rt_arr.shape[1]):
                col_max = rt_arr[:, col].max()
                if col_max > 0:
                    rt_arr[:, col] = rt_arr[:, col] / col_max
            im = ax.imshow(rt_arr, cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(4))
            ax.set_xticklabels(["MAE", "RMSE", "R²", "DirAcc"], fontsize=8)
            ax.set_yticks(range(len(countries_eval)))
            ax.set_yticklabels(countries_eval, fontsize=8)
            ax.set_title("Normalized Metrics Heatmap")
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        fig_comp.savefig(
            f"{PATHS.evaluation_plots}/summary_comparison.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=fig_comp.get_facecolor(),
        )
        plt.close(fig_comp)

        # --- Phase Space Plot (S vs I) ---
        fig_phase, ax_phase = plt.subplots(figsize=(10, 8))
        fig_phase.suptitle("Epidemic Phase Space (S vs I)", fontsize=14, fontweight="bold")
        for country in benchmark_countries:
            country_data_raw = (
                df[df["Country/Region"] == country].sort_values("date").reset_index(drop=True)
            )
            pop_val = country_data_raw["population"].iloc[0]
            total_cases_arr = country_data_raw["total_cases"].values
            new_cases_arr = country_data_raw["new_cases_smoothed"].values
            I_arr = new_cases_arr / max(pop_val, 1) * 100
            S_arr = (pop_val - total_cases_arr) / max(pop_val, 1) * 100
            S_arr = np.clip(S_arr, 0, 100)
            ax_phase.plot(S_arr, I_arr, "-", linewidth=1.5, alpha=0.7, label=country)
        ax_phase.set_xlabel("Susceptible (%)", fontsize=12)
        ax_phase.set_ylabel("Infectious (%)", fontsize=12)
        ax_phase.legend(framealpha=0.3, fontsize=9)
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_facecolor("#1a1d24")
        fig_phase.savefig(
            f"{PATHS.evaluation_plots}/phase_space.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=fig_phase.get_facecolor(),
        )
        plt.close(fig_phase)

        # --- Log-Difference Trend Analysis (GPR Paper Approach) ---
        fig_logdiff, axes_ld = plt.subplots(2, 1, figsize=(14, 10))
        fig_logdiff.suptitle(
            "Log-Difference Trend Analysis (GPR Approach)", fontsize=14, fontweight="bold"
        )

        ax_ld1 = axes_ld[0]
        ax_ld2 = axes_ld[1]

        for country in benchmark_countries:
            country_data_raw = (
                df[df["Country/Region"] == country].sort_values("date").reset_index(drop=True)
            )
            dates = country_data_raw["date"].values

            if "log_diff_7" in country_data_raw.columns:
                log_diff_7 = country_data_raw["log_diff_7"].values
                ax_ld1.plot(dates, log_diff_7, "-", linewidth=1.5, alpha=0.7, label=country)

            if "log_diff_14" in country_data_raw.columns:
                log_diff_14 = country_data_raw["log_diff_14"].values
                ax_ld2.plot(dates, log_diff_14, "-", linewidth=1.5, alpha=0.7, label=country)

        for ax in [ax_ld1, ax_ld2]:
            ax.axhline(y=0, color="white", linestyle="--", alpha=0.5, linewidth=1)
            ax.fill_between(
                ax.get_xlim(),
                0,
                ax.get_ylim()[1],
                alpha=0.05,
                color=COLORS["success"],
                label="Declining",
            )
            ax.fill_between(
                ax.get_xlim(),
                ax.get_ylim()[0],
                0,
                alpha=0.05,
                color=COLORS["secondary"],
                label="Growing",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Δ(t) Log-Difference")
            ax.legend(framealpha=0.3, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        ax_ld1.set_title("η = 7 days (Short-term)")
        ax_ld2.set_title("η = 14 days (Medium-term)")

        fig_logdiff.savefig(
            f"{PATHS.evaluation_plots}/log_diff_trend.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=fig_logdiff.get_facecolor(),
        )
        plt.close(fig_logdiff)

        # --- Wave Detection Analysis ---
        fig_waves, ax_waves = plt.subplots(figsize=(14, 6))
        fig_waves.suptitle("Epidemic Wave Detection", fontsize=14, fontweight="bold")

        for country in benchmark_countries:
            country_data_raw = (
                df[df["Country/Region"] == country].sort_values("date").reset_index(drop=True)
            )
            dates = country_data_raw["date"].values
            cases = country_data_raw["new_cases_smoothed"].values

            # Simple peak detection using local maxima
            peaks, _ = find_peaks(cases, height=np.percentile(cases, 75), distance=30)

            ax_waves.plot(dates, cases, "-", linewidth=1.5, alpha=0.6, label=country)
            if len(peaks) > 0:
                ax_waves.scatter(dates[peaks], cases[peaks], s=50, marker="v", alpha=0.8)

        ax_waves.set_xlabel("Date")
        ax_waves.set_ylabel("New Cases (7d avg)")
        ax_waves.legend(framealpha=0.3, fontsize=9)
        ax_waves.grid(True, alpha=0.3)
        ax_waves.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        fig_waves.savefig(
            f"{PATHS.evaluation_plots}/wave_detection.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=fig_waves.get_facecolor(),
        )
        plt.close(fig_waves)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("\nAverage Metrics:")
    for col in [
        "MAE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "WMAPE",
        "R2",
        "PeakTimingErr",
        "PeakMagErr",
        "DirAccuracy",
    ]:
        if col in results_df.columns:
            print(f"  {col:15s}: {results_df[col].mean():.4f}")

    results_df.to_csv(PATHS.evaluation_results, index=False)

    eval_summary = {
        col.lower(): float(results_df[col].mean()) for col in results_df.columns if col != "Country"
    }
    eval_summary["countries_evaluated"] = list(results_df["Country"])
    with open(PATHS.evaluation_results.replace(".csv", "_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info(f"Results saved to {PATHS.evaluation_results}")


if __name__ == "__main__":
    evaluate()
