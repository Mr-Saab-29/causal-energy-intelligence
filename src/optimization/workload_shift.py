"""Decision rankings for carbon- and cost-aware workload shifting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp_utc"


@dataclass(frozen=True)
class WorkloadConstraints:
    """Feasibility constraints for ranking candidate workload start times."""

    earliest_start_utc: str | None = None
    latest_end_utc: str | None = None
    duration_hours: int = 1
    max_delay_hours: int | None = None
    price_weight: float = 0.5
    carbon_weight: float = 0.5
    methodology: str = "direct_operational_emissions"


def run_workload_decision_ranking(
    price_rankings_path: str | Path = "reports/rankings/price_decision_rankings.csv",
    carbon_intensity_path: str | Path = "reports/carbon/hourly_carbon_intensity.csv",
    ranking_output_path: str | Path = "reports/rankings/workload_decision_rankings.csv",
    recommendation_output_path: str | Path = (
        "reports/recommendations/top5_workload_recommendations.csv"
    ),
    metrics_output_path: str | Path = "reports/metrics/workload_decision_metrics.json",
    constraints: WorkloadConstraints | None = None,
    top_n_recommendations: int = 5,
) -> dict[str, Any]:
    """Build combined cost/carbon workload rankings from saved forecast artifacts."""
    constraints = constraints or WorkloadConstraints()
    hourly = load_hourly_decision_inputs(price_rankings_path, carbon_intensity_path, constraints)
    rankings = build_workload_decision_rankings(hourly, constraints)
    recommendations = build_top_workload_recommendations(rankings, top_n=top_n_recommendations)
    metrics = summarize_workload_decision_metrics(rankings)
    write_csv(ranking_output_path, rankings)
    write_csv(recommendation_output_path, recommendations)
    write_json(
        metrics_output_path,
        {
            "constraints": asdict(constraints),
            "top_n_recommendations": top_n_recommendations,
            "summary": metrics,
        },
    )
    return {"constraints": asdict(constraints), "summary": metrics}


def load_hourly_decision_inputs(
    price_rankings_path: str | Path,
    carbon_intensity_path: str | Path,
    constraints: WorkloadConstraints,
) -> pd.DataFrame:
    """Join hourly price ranking signals with hourly carbon-intensity signals."""
    prices = pd.read_csv(price_rankings_path, parse_dates=[TIMESTAMP_COLUMN])
    carbon = pd.read_csv(carbon_intensity_path, parse_dates=[TIMESTAMP_COLUMN])
    carbon = carbon[carbon["methodology"] == constraints.methodology].copy()
    if carbon.empty:
        raise ValueError(f"No carbon rows found for methodology {constraints.methodology!r}")

    carbon_columns = [
        TIMESTAMP_COLUMN,
        "window",
        "model",
        "predicted_carbon_intensity_g_co2e_per_kwh",
        "actual_carbon_intensity_g_co2e_per_kwh",
        "predicted_total_emissions_kg_co2e",
        "actual_total_emissions_kg_co2e",
    ]
    merged = prices.merge(
        carbon[carbon_columns],
        on=[TIMESTAMP_COLUMN, "window", "model"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping price and carbon rows found")

    merged[TIMESTAMP_COLUMN] = pd.to_datetime(merged[TIMESTAMP_COLUMN], utc=True)
    return merged.sort_values(["model", "window", TIMESTAMP_COLUMN]).reset_index(drop=True)


def build_workload_decision_rankings(
    hourly: pd.DataFrame,
    constraints: WorkloadConstraints,
) -> pd.DataFrame:
    """Create feasible workload candidates and rank them by combined predicted score."""
    if constraints.duration_hours < 1:
        raise ValueError("duration_hours must be at least 1")
    if constraints.price_weight < 0 or constraints.carbon_weight < 0:
        raise ValueError("price_weight and carbon_weight must be non-negative")
    if constraints.price_weight == 0 and constraints.carbon_weight == 0:
        raise ValueError("At least one of price_weight or carbon_weight must be positive")

    hourly = apply_time_constraints(hourly, constraints)
    candidates = build_candidate_windows(
        hourly,
        constraints.duration_hours,
        constrained_window=bool(constraints.earliest_start_utc or constraints.latest_end_utc),
    )
    if candidates.empty:
        raise ValueError("No feasible workload candidates after applying constraints")

    candidates["decision_group"] = decision_group_id(candidates, constraints)
    group_columns = ["window", "model", "decision_group"]
    candidates["predicted_price_rank"] = rank_within_group(
        candidates,
        group_columns,
        "predicted_avg_price_eur_mwh",
    )
    candidates["predicted_carbon_rank"] = rank_within_group(
        candidates,
        group_columns,
        "predicted_avg_carbon_intensity_g_co2e_per_kwh",
    )
    candidates["actual_price_rank"] = rank_within_group(
        candidates,
        group_columns,
        "actual_avg_price_eur_mwh",
    )
    candidates["actual_carbon_rank"] = rank_within_group(
        candidates,
        group_columns,
        "actual_avg_carbon_intensity_g_co2e_per_kwh",
    )
    candidates["candidate_count"] = candidates.groupby(group_columns)[TIMESTAMP_COLUMN].transform("size")
    candidates["predicted_price_rank_pct"] = normalized_rank(
        candidates["predicted_price_rank"],
        candidates["candidate_count"],
    )
    candidates["predicted_carbon_rank_pct"] = normalized_rank(
        candidates["predicted_carbon_rank"],
        candidates["candidate_count"],
    )
    candidates["actual_price_rank_pct"] = normalized_rank(
        candidates["actual_price_rank"],
        candidates["candidate_count"],
    )
    candidates["actual_carbon_rank_pct"] = normalized_rank(
        candidates["actual_carbon_rank"],
        candidates["candidate_count"],
    )
    weight_sum = constraints.price_weight + constraints.carbon_weight
    price_weight = constraints.price_weight / weight_sum
    carbon_weight = constraints.carbon_weight / weight_sum
    candidates["predicted_combined_score"] = (
        price_weight * candidates["predicted_price_rank_pct"]
        + carbon_weight * candidates["predicted_carbon_rank_pct"]
    )
    candidates["actual_combined_score"] = (
        price_weight * candidates["actual_price_rank_pct"]
        + carbon_weight * candidates["actual_carbon_rank_pct"]
    )
    candidates["predicted_decision_rank"] = rank_within_group(
        candidates,
        group_columns,
        "predicted_combined_score",
    )
    candidates["actual_decision_rank"] = rank_within_group(
        candidates,
        group_columns,
        "actual_combined_score",
    )

    annotate_regret_and_savings(candidates, group_columns)
    sort_columns = group_columns + ["predicted_decision_rank", TIMESTAMP_COLUMN]
    return candidates.sort_values(sort_columns).reset_index(drop=True)


def apply_time_constraints(hourly: pd.DataFrame, constraints: WorkloadConstraints) -> pd.DataFrame:
    """Filter hourly rows by absolute workload feasibility constraints."""
    frame = hourly.copy()
    if constraints.earliest_start_utc:
        earliest = pd.Timestamp(constraints.earliest_start_utc, tz="UTC")
        frame = frame[frame[TIMESTAMP_COLUMN] >= earliest]
    if constraints.latest_end_utc:
        latest_start = pd.Timestamp(constraints.latest_end_utc, tz="UTC") - pd.Timedelta(
            hours=constraints.duration_hours
        )
        frame = frame[frame[TIMESTAMP_COLUMN] <= latest_start]
    if constraints.earliest_start_utc and constraints.max_delay_hours is not None:
        latest_delay_start = pd.Timestamp(
            constraints.earliest_start_utc,
            tz="UTC",
        ) + pd.Timedelta(hours=constraints.max_delay_hours)
        frame = frame[frame[TIMESTAMP_COLUMN] <= latest_delay_start]
    return frame


def build_candidate_windows(
    hourly: pd.DataFrame,
    duration_hours: int,
    constrained_window: bool = False,
) -> pd.DataFrame:
    """Aggregate contiguous hourly rows into feasible workload windows."""
    records: list[dict[str, Any]] = []
    group_columns = ["window", "model"] if constrained_window else ["window", "model", "decision_date"]
    for group_key, group in hourly.groupby(group_columns, observed=True):
        if constrained_window:
            window, model = group_key
        else:
            window, model, grouped_decision_date = group_key
        group = group.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        for start_index in range(0, len(group) - duration_hours + 1):
            candidate = group.iloc[start_index : start_index + duration_hours]
            timestamps = candidate[TIMESTAMP_COLUMN]
            if not is_contiguous_hourly(timestamps):
                continue
            start_time = timestamps.iloc[0]
            end_time = timestamps.iloc[-1] + pd.Timedelta(hours=1)
            decision_date = (
                start_time.date().isoformat() if constrained_window else grouped_decision_date
            )
            records.append(
                {
                    TIMESTAMP_COLUMN: start_time,
                    "workload_end_utc": end_time,
                    "window": window,
                    "model": model,
                    "decision_date": decision_date,
                    "duration_hours": duration_hours,
                    "actual_avg_price_eur_mwh": candidate["actual_price_eur_mwh"].mean(),
                    "predicted_avg_price_eur_mwh": candidate["predicted_price_eur_mwh"].mean(),
                    "actual_avg_carbon_intensity_g_co2e_per_kwh": candidate[
                        "actual_carbon_intensity_g_co2e_per_kwh"
                    ].mean(),
                    "predicted_avg_carbon_intensity_g_co2e_per_kwh": candidate[
                        "predicted_carbon_intensity_g_co2e_per_kwh"
                    ].mean(),
                    "actual_total_emissions_kg_co2e": candidate[
                        "actual_total_emissions_kg_co2e"
                    ].sum(),
                    "predicted_total_emissions_kg_co2e": candidate[
                        "predicted_total_emissions_kg_co2e"
                    ].sum(),
                }
            )
    return pd.DataFrame(records)


def is_contiguous_hourly(timestamps: pd.Series) -> bool:
    """Return whether timestamps form a contiguous hourly block."""
    if len(timestamps) <= 1:
        return True
    deltas = timestamps.diff().dropna()
    return bool((deltas == pd.Timedelta(hours=1)).all())


def decision_group_id(frame: pd.DataFrame, constraints: WorkloadConstraints) -> pd.Series:
    """Return the decision group identifier for candidate comparisons."""
    if constraints.earliest_start_utc or constraints.latest_end_utc:
        return pd.Series("constrained_window", index=frame.index)
    return frame["decision_date"]


def rank_within_group(frame: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.Series:
    """Rank ascending values within decision groups."""
    return (
        frame.groupby(group_columns, observed=True)[value_column]
        .rank(method="first", ascending=True)
        .astype(int)
    )


def normalized_rank(rank: pd.Series, candidate_count: pd.Series) -> pd.Series:
    """Normalize ranks to [0, 1], preserving zero for one-candidate groups."""
    denominator = (candidate_count - 1).replace(0, np.nan)
    return ((rank - 1) / denominator).fillna(0.0)


def annotate_regret_and_savings(frame: pd.DataFrame, group_columns: list[str]) -> None:
    """Add regret and savings metrics against actual best and run-now baselines."""
    frame["actual_best_combined_score"] = frame.groupby(group_columns, observed=True)[
        "actual_combined_score"
    ].transform("min")
    frame["actual_best_price_eur_mwh"] = frame.groupby(group_columns, observed=True)[
        "actual_avg_price_eur_mwh"
    ].transform("min")
    frame["actual_best_carbon_intensity_g_co2e_per_kwh"] = frame.groupby(
        group_columns,
        observed=True,
    )["actual_avg_carbon_intensity_g_co2e_per_kwh"].transform("min")
    run_now = (
        frame.sort_values(TIMESTAMP_COLUMN)
        .groupby(group_columns, observed=True)
        .head(1)
        .set_index(group_columns)
    )
    frame_index = pd.MultiIndex.from_frame(frame[group_columns])
    frame["run_now_price_eur_mwh"] = run_now["actual_avg_price_eur_mwh"].reindex(frame_index).to_numpy()
    frame["run_now_carbon_intensity_g_co2e_per_kwh"] = run_now[
        "actual_avg_carbon_intensity_g_co2e_per_kwh"
    ].reindex(frame_index).to_numpy()
    frame["combined_regret"] = frame["actual_combined_score"] - frame["actual_best_combined_score"]
    frame["cost_regret_eur_mwh"] = frame["actual_avg_price_eur_mwh"] - frame["actual_best_price_eur_mwh"]
    frame["carbon_regret_g_co2e_per_kwh"] = (
        frame["actual_avg_carbon_intensity_g_co2e_per_kwh"]
        - frame["actual_best_carbon_intensity_g_co2e_per_kwh"]
    )
    frame["cost_savings_vs_run_now_eur_mwh"] = (
        frame["run_now_price_eur_mwh"] - frame["actual_avg_price_eur_mwh"]
    )
    frame["carbon_savings_vs_run_now_g_co2e_per_kwh"] = (
        frame["run_now_carbon_intensity_g_co2e_per_kwh"]
        - frame["actual_avg_carbon_intensity_g_co2e_per_kwh"]
    )
    frame["is_predicted_best"] = frame["predicted_decision_rank"] == 1
    frame["is_actual_best"] = frame["actual_decision_rank"] == 1
    frame["is_predicted_top_3"] = frame["predicted_decision_rank"] <= 3
    frame["is_actual_top_3"] = frame["actual_decision_rank"] <= 3


def summarize_workload_decision_metrics(rankings: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize decision-ranking quality by model."""
    summaries: list[dict[str, Any]] = []
    for model, model_frame in rankings.groupby("model", observed=True):
        group_columns = ["window", "decision_group"]
        predicted_best = model_frame[model_frame["predicted_decision_rank"] == 1]
        actual_best = model_frame[model_frame["actual_decision_rank"] == 1]
        top_3_capture = (
            actual_best.groupby(group_columns, observed=True)["is_predicted_top_3"].any().mean()
        )
        summaries.append(
            {
                "model": model,
                "decision_groups": int(model_frame.groupby(group_columns, observed=True).ngroups),
                "top_1_hit_rate": float(predicted_best["is_actual_best"].mean()),
                "top_3_capture_rate": float(top_3_capture),
                "mean_combined_regret": float(predicted_best["combined_regret"].mean()),
                "mean_cost_regret_eur_mwh": float(predicted_best["cost_regret_eur_mwh"].mean()),
                "mean_carbon_regret_g_co2e_per_kwh": float(
                    predicted_best["carbon_regret_g_co2e_per_kwh"].mean()
                ),
                "mean_cost_savings_vs_run_now_eur_mwh": float(
                    predicted_best["cost_savings_vs_run_now_eur_mwh"].mean()
                ),
                "mean_carbon_savings_vs_run_now_g_co2e_per_kwh": float(
                    predicted_best["carbon_savings_vs_run_now_g_co2e_per_kwh"].mean()
                ),
                "mean_actual_rank_of_recommendation": float(
                    predicted_best["actual_decision_rank"].mean()
                ),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_combined_regret"])


def build_top_workload_recommendations(rankings: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top-N recommended workload start times per model/window/decision group."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    group_columns = ["window", "model", "decision_group"]
    recommended = (
        rankings[rankings["predicted_decision_rank"] <= top_n]
        .sort_values(group_columns + ["predicted_decision_rank", TIMESTAMP_COLUMN])
        .copy()
    )
    recommended["recommendation_rank"] = recommended["predicted_decision_rank"]
    recommendation_columns = [
        "window",
        "model",
        "decision_group",
        "recommendation_rank",
        TIMESTAMP_COLUMN,
        "workload_end_utc",
        "duration_hours",
        "predicted_combined_score",
        "predicted_avg_price_eur_mwh",
        "predicted_avg_carbon_intensity_g_co2e_per_kwh",
        "predicted_total_emissions_kg_co2e",
        "predicted_price_rank",
        "predicted_carbon_rank",
        "candidate_count",
        "actual_decision_rank",
        "combined_regret",
        "cost_regret_eur_mwh",
        "carbon_regret_g_co2e_per_kwh",
        "cost_savings_vs_run_now_eur_mwh",
        "carbon_savings_vs_run_now_g_co2e_per_kwh",
    ]
    recommended = recommended[recommendation_columns]
    float_columns = recommended.select_dtypes(include=["float"]).columns
    recommended[float_columns] = recommended[float_columns].round(2)
    return recommended.reset_index(drop=True)


def optimize_workload_shift(
    forecast_rows: list[dict[str, object]],
    max_shift_hours: int = 6,
) -> dict[str, object]:
    """Choose the best row from already-ranked workload forecast rows."""
    if not forecast_rows:
        return {
            "max_shift_hours": max_shift_hours,
            "input_rows": 0,
            "recommended_shift_hours": 0,
        }
    ranked = sorted(forecast_rows, key=lambda row: row.get("predicted_decision_rank", float("inf")))
    return {
        "max_shift_hours": max_shift_hours,
        "input_rows": len(forecast_rows),
        "recommended_shift_hours": 0,
        "recommendation": ranked[0],
    }


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV file, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
