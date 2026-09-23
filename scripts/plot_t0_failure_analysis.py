"""Plot the saved final-week t0 consumption failure analysis for the README."""

from __future__ import annotations

import argparse
from datetime import UTC
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


DEFAULT_PREDICTIONS = Path(
    "reports/benchmarks/t0_2026_05_26_to_08_23_v3/experiments_v1/predictions.csv.gz"
)
DEFAULT_OUTPUT = Path("docs/assets/t0-failure-analysis.png")
MODELS = ["lightgbm", "lightgbm_quantile", "t0-alpha"]
LABELS = {
    "lightgbm": "LightGBM",
    "lightgbm_quantile": "Quantile LightGBM",
    "t0-alpha": "t0-alpha",
}
COLORS = {
    "lightgbm": "#4B5563",
    "lightgbm_quantile": "#D97706",
    "t0-alpha": "#2563EB",
}


def load_final_week(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    final_origin = frame["origin"].max()
    final_week = frame[
        (frame["origin"] >= final_origin - pd.Timedelta(days=6))
        & (frame["target"] == "consumption_mwh")
        & (frame["variant"] == "calendar_30d")
        & frame["model"].isin(MODELS)
    ].copy()
    counts = final_week.groupby("model")["origin"].nunique()
    if counts.reindex(MODELS).tolist() != [7, 7, 7]:
        raise ValueError(f"Expected seven final-week origins per model; found {counts.to_dict()}")
    return final_week


def plot_failure_analysis(predictions: Path, output: Path) -> None:
    frame = load_final_week(predictions)
    t0 = frame[
        (frame["model"] == "t0-alpha")
        & frame["origin"].dt.date.astype(str).isin(["2026-08-20", "2026-08-21", "2026-08-22"])
    ].sort_values("timestamp_utc")
    if len(t0) != 72 or not t0[["actual", "q0.1", "q0.5", "q0.9"]].notna().all().all():
        raise ValueError("Expected 72 complete t0 rows for August 20–22")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "timezone": "UTC",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.7,
        }
    )
    figure, (forecast_ax, horizon_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8.5),
        gridspec_kw={"height_ratios": [1.35, 1]},
        layout="constrained",
    )
    figure.set_facecolor("white")
    figure.suptitle(
        "Late-horizon degradation of t0-alpha on final-week consumption forecasts",
        fontsize=16,
        fontweight="bold",
    )

    forecast_ax.fill_between(
        t0["timestamp_utc"],
        t0["q0.1"],
        t0["q0.9"],
        color="#93C5FD",
        alpha=0.38,
        linewidth=0,
        label="t0 q10–q90 interval",
        zorder=1,
    )
    forecast_ax.plot(
        t0["timestamp_utc"],
        t0["actual"],
        color="#111827",
        linewidth=2.2,
        label="Actual consumption",
        zorder=4,
    )
    forecast_ax.plot(
        t0["timestamp_utc"],
        t0["q0.5"],
        color=COLORS["t0-alpha"],
        linewidth=2,
        label="t0 median (q50)",
        zorder=3,
    )
    for origin in sorted(t0["origin"].unique()):
        origin = pd.Timestamp(origin)
        split = origin + pd.Timedelta(hours=12)
        end = origin + pd.Timedelta(hours=24)
        forecast_ax.axvspan(split, end, color="#F3F4F6", alpha=0.8, zorder=0)
        forecast_ax.axvline(split, color="#9CA3AF", linestyle="--", linewidth=1, zorder=2)
        if origin > t0["origin"].min():
            forecast_ax.axvline(origin, color="#D1D5DB", linewidth=0.8, zorder=2)
    forecast_ax.set_title("A. Forecast behaviour on the clearest failure dates", loc="left", fontsize=12)
    forecast_ax.set_ylabel("French electricity consumption (MWh)")
    forecast_ax.set_xlim(t0["timestamp_utc"].min(), t0["timestamp_utc"].max())
    forecast_ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=UTC))
    forecast_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=UTC))
    forecast_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    forecast_ax.grid(axis="y", alpha=0.75)
    forecast_ax.legend(loc="upper left", frameon=False, ncols=3)
    forecast_ax.text(
        0.995,
        0.03,
        "Gray bands: forecast hours 13–24",
        transform=forecast_ax.transAxes,
        ha="right",
        va="bottom",
        color="#4B5563",
        fontsize=9,
    )

    frame["absolute_error"] = (frame["prediction"] - frame["actual"]).abs()
    lead_mae = frame.groupby(["model", "lead_hour"], as_index=False)["absolute_error"].mean()
    for model in MODELS:
        values = lead_mae[lead_mae["model"] == model]
        horizon_ax.plot(
            values["lead_hour"],
            values["absolute_error"],
            color=COLORS[model],
            linewidth=2,
            marker="o",
            markersize=4,
            label=LABELS[model],
        )
    horizon_ax.axvline(12.5, color="#6B7280", linestyle="--", linewidth=1.2)
    horizon_ax.text(12.8, horizon_ax.get_ylim()[1] * 0.96, "12h split", color="#4B5563", va="top")
    horizon_ax.set_title("B. MAE by forecast lead across all seven final-week origins", loc="left", fontsize=12)
    horizon_ax.set_xlabel("Forecast lead (hours)")
    horizon_ax.set_ylabel("MAE (MWh)")
    horizon_ax.set_xlim(1, 24)
    horizon_ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    horizon_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.1f}k"))
    horizon_ax.grid(alpha=0.75)
    horizon_ax.legend(loc="upper left", frameon=False, ncols=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_failure_analysis(args.predictions, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
