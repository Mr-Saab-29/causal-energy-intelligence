"""Decision rankings for carbon- and cost-aware workload shifting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp_utc"
DEFAULT_SCENARIOS = [
    {"scenario": "clean_first", "price_weight": 0.2, "carbon_weight": 0.8},
    {"scenario": "balanced", "price_weight": 0.5, "carbon_weight": 0.5},
    {"scenario": "cost_aware_clean", "price_weight": 0.4, "carbon_weight": 0.6},
]


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
    price_metrics_path: str | Path = "reports/metrics/price_baseline_metrics.json",
    carbon_intensity_path: str | Path = "reports/carbon/hourly_carbon_intensity.csv",
    carbon_metrics_path: str | Path = "reports/metrics/carbon_forecast_metrics.json",
    ranking_output_path: str | Path = "reports/rankings/workload_decision_rankings.csv",
    recommendation_output_path: str | Path = (
        "reports/recommendations/top5_workload_recommendations.csv"
    ),
    champion_recommendation_output_path: str | Path = (
        "reports/recommendations/champion_workload_recommendations.csv"
    ),
    metrics_output_path: str | Path = "reports/metrics/workload_decision_metrics.json",
    ranking_specific_metrics_path: str | Path = (
        "reports/metrics/ranking_specific_metrics.json"
    ),
    champion_output_path: str | Path = "reports/metrics/champion_model_selection.json",
    scenario_recommendation_output_path: str | Path = (
        "reports/scenarios/workload_scenario_recommendations.csv"
    ),
    scenario_metrics_output_path: str | Path = "reports/metrics/scenario_reranking_metrics.json",
    constraints: WorkloadConstraints | None = None,
    top_n_recommendations: int = 5,
) -> dict[str, Any]:
    """Build combined cost/carbon workload rankings from saved forecast artifacts."""
    constraints = constraints or WorkloadConstraints()
    hourly = load_hourly_decision_inputs(price_rankings_path, carbon_intensity_path, constraints)
    rankings = build_workload_decision_rankings(hourly, constraints)
    recommendations = build_top_workload_recommendations(rankings, top_n=top_n_recommendations)
    recommendations = add_recommendation_confidence(recommendations, rankings, top_n_recommendations)
    metrics = summarize_workload_decision_metrics(rankings)
    ranking_specific_metrics = summarize_ranking_specific_metrics(rankings)
    champion = select_champion_model(
        price_metrics_path=price_metrics_path,
        carbon_metrics_path=carbon_metrics_path,
        ranking_specific_metrics=ranking_specific_metrics["summary"],
        methodology=constraints.methodology,
    )
    scenario_recommendations, scenario_metrics = build_scenario_rerankings(
        rankings,
        top_n=top_n_recommendations,
    )
    champion_recommendations = build_champion_workload_recommendations(
        recommendations,
        champion["champion_model"],
    )
    write_csv(ranking_output_path, rankings)
    write_csv(recommendation_output_path, recommendations)
    write_csv(champion_recommendation_output_path, champion_recommendations)
    write_csv(scenario_recommendation_output_path, scenario_recommendations)
    write_json(
        metrics_output_path,
        {
            "constraints": asdict(constraints),
            "top_n_recommendations": top_n_recommendations,
            "summary": metrics,
        },
    )
    write_json(ranking_specific_metrics_path, ranking_specific_metrics)
    write_json(champion_output_path, champion)
    write_json(scenario_metrics_output_path, scenario_metrics)
    return {
        "constraints": asdict(constraints),
        "summary": metrics,
        "champion_model": champion["champion_model"],
    }


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
    merged = merged.sort_values(["model", "window", TIMESTAMP_COLUMN]).reset_index(drop=True)
    merged["previous_day_price_eur_mwh"] = merged.groupby(
        ["window", "model"],
        observed=True,
    )["actual_price_eur_mwh"].shift(24)
    return merged


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
    annotate_price_direction(candidates)
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
                    "previous_day_avg_price_eur_mwh": candidate[
                        "previous_day_price_eur_mwh"
                    ].mean(),
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


def annotate_price_direction(frame: pd.DataFrame) -> None:
    """Add price direction labels versus the previous day at the same time."""
    frame["predicted_price_change_vs_previous_day_eur_mwh"] = (
        frame["predicted_avg_price_eur_mwh"] - frame["previous_day_avg_price_eur_mwh"]
    )
    frame["actual_price_change_vs_previous_day_eur_mwh"] = (
        frame["actual_avg_price_eur_mwh"] - frame["previous_day_avg_price_eur_mwh"]
    )
    frame["predicted_price_direction_vs_previous_day"] = frame[
        "predicted_price_change_vs_previous_day_eur_mwh"
    ].map(price_direction_label)
    frame["actual_price_direction_vs_previous_day"] = frame[
        "actual_price_change_vs_previous_day_eur_mwh"
    ].map(price_direction_label)
    frame["price_direction_correct"] = (
        frame["predicted_price_direction_vs_previous_day"]
        == frame["actual_price_direction_vs_previous_day"]
    ).astype(float)
    frame.loc[
        frame["previous_day_avg_price_eur_mwh"].isna(),
        "price_direction_correct",
    ] = np.nan


def price_direction_label(change: float | None) -> str:
    """Convert a price delta into a dashboard-friendly direction label."""
    if pd.isna(change):
        return "unknown"
    if change > 0:
        return "increase"
    if change < 0:
        return "decrease"
    return "flat"


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
        "predicted_price_direction_vs_previous_day",
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


def add_recommendation_confidence(
    recommendations: pd.DataFrame,
    rankings: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Add model-agreement and score-margin based confidence fields."""
    output = recommendations.copy()
    agreement_counts = (
        recommendations.groupby(["window", "decision_group", TIMESTAMP_COLUMN], observed=True)[
            "model"
        ]
        .nunique()
        .rename("agreement_count")
    )
    model_counts = (
        rankings.groupby(["window", "decision_group"], observed=True)["model"]
        .nunique()
        .rename("model_count")
    )
    output_index = pd.MultiIndex.from_frame(output[["window", "decision_group", TIMESTAMP_COLUMN]])
    group_index = pd.MultiIndex.from_frame(output[["window", "decision_group"]])
    output["agreement_count"] = agreement_counts.reindex(output_index).to_numpy()
    output["model_count"] = model_counts.reindex(group_index).to_numpy()
    output["model_agreement_rate"] = output["agreement_count"] / output["model_count"]
    rank_denominator = max(top_n - 1, 1)
    output["rank_confidence_component"] = 1 - (
        (output["recommendation_rank"] - 1) / rank_denominator
    )

    score_margins = calculate_score_margin_components(rankings)
    margin_index = pd.MultiIndex.from_frame(
        output[["window", "model", "decision_group", TIMESTAMP_COLUMN]]
    )
    output["score_margin_component"] = score_margins.reindex(margin_index).fillna(0).to_numpy()
    output["confidence_score"] = (
        0.45 * output["rank_confidence_component"]
        + 0.35 * output["model_agreement_rate"]
        + 0.20 * output["score_margin_component"]
    )
    output["confidence_level"] = output["confidence_score"].map(confidence_label)
    float_columns = output.select_dtypes(include=["float"]).columns
    output[float_columns] = output[float_columns].round(2)
    return output


def calculate_score_margin_components(rankings: pd.DataFrame) -> pd.Series:
    """Calculate normalized score separation from the next-best candidate."""
    frame = rankings.sort_values(
        ["window", "model", "decision_group", "predicted_decision_rank"],
    ).copy()
    group_columns = ["window", "model", "decision_group"]
    frame["next_score"] = frame.groupby(group_columns, observed=True)[
        "predicted_combined_score"
    ].shift(-1)
    frame["score_range"] = frame.groupby(group_columns, observed=True)[
        "predicted_combined_score"
    ].transform(lambda values: values.max() - values.min())
    frame["score_margin_component"] = (
        (frame["next_score"] - frame["predicted_combined_score"]) / frame["score_range"]
    ).clip(lower=0, upper=1)
    frame["score_margin_component"] = frame["score_margin_component"].fillna(0)
    return frame.set_index(group_columns + [TIMESTAMP_COLUMN])["score_margin_component"]


def confidence_label(score: float) -> str:
    """Map confidence score to a compact product-facing label."""
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def build_champion_workload_recommendations(
    recommendations: pd.DataFrame,
    champion_model: str | None,
) -> pd.DataFrame:
    """Filter recommendations to the dynamically selected champion model."""
    if not champion_model:
        return recommendations.iloc[0:0].copy()
    return recommendations[recommendations["model"] == champion_model].reset_index(drop=True)


def summarize_ranking_specific_metrics(rankings: pd.DataFrame) -> dict[str, Any]:
    """Evaluate pairwise loss, top-5 classification, and regret by decision group."""
    summary: list[dict[str, Any]] = []
    by_decision_group: list[dict[str, Any]] = []
    group_columns = ["window", "model", "decision_group"]

    for (window, model, decision_group), group in rankings.groupby(group_columns, observed=True):
        group = group.sort_values("predicted_decision_rank").copy()
        pairwise_loss = calculate_pairwise_ranking_loss(
            predicted=group["predicted_combined_score"].to_numpy(dtype=float),
            actual=group["actual_combined_score"].to_numpy(dtype=float),
        )
        classification = calculate_top_k_classification(group, top_k=5)
        predicted_best = group[group["predicted_decision_rank"] == 1].iloc[0]
        by_decision_group.append(
            {
                "window": window,
                "model": model,
                "decision_group": decision_group,
                "pairwise_ranking_loss": pairwise_loss,
                **classification,
                "price_direction_accuracy": safe_mean(group["price_direction_correct"]),
                "top_1_combined_regret": float(predicted_best["combined_regret"]),
                "top_1_carbon_regret_g_co2e_per_kwh": float(
                    predicted_best["carbon_regret_g_co2e_per_kwh"]
                ),
                "top_1_cost_regret_eur_mwh": float(predicted_best["cost_regret_eur_mwh"]),
            }
        )

    group_frame = pd.DataFrame(by_decision_group)
    for model, model_frame in group_frame.groupby("model", observed=True):
        summary.append(
            {
                "model": model,
                "decision_groups": int(len(model_frame)),
                "pairwise_ranking_loss": float(model_frame["pairwise_ranking_loss"].mean()),
                "top_5_precision": float(model_frame["top_5_precision"].mean()),
                "top_5_recall": float(model_frame["top_5_recall"].mean()),
                "top_5_f1": float(model_frame["top_5_f1"].mean()),
                "price_direction_accuracy": float(model_frame["price_direction_accuracy"].mean()),
                "price_direction_error": float(1 - model_frame["price_direction_accuracy"].mean()),
                "mean_top_1_combined_regret": float(
                    model_frame["top_1_combined_regret"].mean()
                ),
                "mean_top_1_carbon_regret_g_co2e_per_kwh": float(
                    model_frame["top_1_carbon_regret_g_co2e_per_kwh"].mean()
                ),
                "mean_top_1_cost_regret_eur_mwh": float(
                    model_frame["top_1_cost_regret_eur_mwh"].mean()
                ),
            }
        )

    return {
        "summary": sorted(summary, key=lambda row: row["pairwise_ranking_loss"]),
        "by_decision_group": by_decision_group,
    }


def calculate_pairwise_ranking_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Return pairwise disagreement rate between predicted and actual ordering."""
    compared = 0
    mistakes = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            actual_order = np.sign(actual[left] - actual[right])
            if actual_order == 0:
                continue
            predicted_order = np.sign(predicted[left] - predicted[right])
            compared += 1
            mistakes += int(predicted_order != actual_order)
    if compared == 0:
        return 0.0
    return float(mistakes / compared)


def calculate_top_k_classification(group: pd.DataFrame, top_k: int) -> dict[str, float]:
    """Evaluate top-k recommendation as a classification task."""
    predicted_top = group["predicted_decision_rank"] <= top_k
    actual_top = group["actual_decision_rank"] <= top_k
    true_positive = int((predicted_top & actual_top).sum())
    predicted_positive = int(predicted_top.sum())
    actual_positive = int(actual_top.sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        f"top_{top_k}_precision": float(precision),
        f"top_{top_k}_recall": float(recall),
        f"top_{top_k}_f1": float(f1),
    }


def build_scenario_rerankings(
    rankings: pd.DataFrame,
    scenarios: list[dict[str, Any]] | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rerank candidates under alternative clean-hour decision preferences."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    scenario_frames: list[pd.DataFrame] = []
    scenario_metrics: list[dict[str, Any]] = []
    group_columns = ["window", "model", "decision_group"]

    for scenario in scenarios:
        scenario_frame = rankings.copy()
        weight_sum = scenario["price_weight"] + scenario["carbon_weight"]
        price_weight = scenario["price_weight"] / weight_sum
        carbon_weight = scenario["carbon_weight"] / weight_sum
        scenario_frame["scenario"] = scenario["scenario"]
        scenario_frame["scenario_price_weight"] = price_weight
        scenario_frame["scenario_carbon_weight"] = carbon_weight
        scenario_frame["predicted_scenario_score"] = (
            price_weight * scenario_frame["predicted_price_rank_pct"]
            + carbon_weight * scenario_frame["predicted_carbon_rank_pct"]
        )
        scenario_frame["actual_scenario_score"] = (
            price_weight * scenario_frame["actual_price_rank_pct"]
            + carbon_weight * scenario_frame["actual_carbon_rank_pct"]
        )
        scenario_group_columns = ["scenario", *group_columns]
        scenario_frame["predicted_scenario_rank"] = rank_within_group(
            scenario_frame,
            scenario_group_columns,
            "predicted_scenario_score",
        )
        scenario_frame["actual_scenario_rank"] = rank_within_group(
            scenario_frame,
            scenario_group_columns,
            "actual_scenario_score",
        )
        scenario_frame["actual_best_scenario_score"] = scenario_frame.groupby(
            scenario_group_columns,
            observed=True,
        )["actual_scenario_score"].transform("min")
        scenario_frame["scenario_regret"] = (
            scenario_frame["actual_scenario_score"]
            - scenario_frame["actual_best_scenario_score"]
        )
        scenario_frames.append(
            build_top_scenario_recommendations(scenario_frame, top_n=top_n)
        )
        scenario_metrics.extend(summarize_scenario_metrics(scenario_frame))

    return (
        pd.concat(scenario_frames, ignore_index=True),
        {"scenarios": scenarios, "summary": scenario_metrics},
    )


def build_top_scenario_recommendations(rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return top-N recommendations for one reranked scenario frame."""
    recommended = rankings[rankings["predicted_scenario_rank"] <= top_n].copy()
    recommended = recommended.sort_values(
        ["scenario", "window", "model", "decision_group", "predicted_scenario_rank"],
    )
    recommended["recommendation_rank"] = recommended["predicted_scenario_rank"]
    columns = [
        "scenario",
        "scenario_price_weight",
        "scenario_carbon_weight",
        "window",
        "model",
        "decision_group",
        "recommendation_rank",
        TIMESTAMP_COLUMN,
        "workload_end_utc",
        "duration_hours",
        "predicted_scenario_score",
        "predicted_price_direction_vs_previous_day",
        "predicted_avg_carbon_intensity_g_co2e_per_kwh",
        "predicted_total_emissions_kg_co2e",
        "predicted_carbon_rank",
        "actual_scenario_rank",
        "scenario_regret",
        "carbon_regret_g_co2e_per_kwh",
        "carbon_savings_vs_run_now_g_co2e_per_kwh",
    ]
    recommended = recommended[columns]
    float_columns = recommended.select_dtypes(include=["float"]).columns
    recommended[float_columns] = recommended[float_columns].round(2)
    return recommended.reset_index(drop=True)


def summarize_scenario_metrics(rankings: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize reranked scenario quality by model."""
    summaries: list[dict[str, Any]] = []
    group_columns = ["scenario", "window", "decision_group"]
    for (scenario, model), model_frame in rankings.groupby(["scenario", "model"], observed=True):
        predicted_best = model_frame[model_frame["predicted_scenario_rank"] == 1]
        actual_best = model_frame[model_frame["actual_scenario_rank"] == 1]
        top_3_capture = (
            actual_best.groupby(group_columns, observed=True)
            .apply(
                lambda group: bool(
                    model_frame.loc[group.index, "predicted_scenario_rank"].le(3).iloc[0]
                ),
                include_groups=False,
            )
            .mean()
        )
        summaries.append(
            {
                "scenario": scenario,
                "model": model,
                "decision_groups": int(
                    model_frame.groupby(group_columns, observed=True).ngroups
                ),
                "top_1_hit_rate": float((predicted_best["actual_scenario_rank"] == 1).mean()),
                "top_3_capture_rate": float(top_3_capture),
                "mean_scenario_regret": float(predicted_best["scenario_regret"].mean()),
                "mean_carbon_regret_g_co2e_per_kwh": float(
                    predicted_best["carbon_regret_g_co2e_per_kwh"].mean()
                ),
                "mean_carbon_savings_vs_run_now_g_co2e_per_kwh": float(
                    predicted_best["carbon_savings_vs_run_now_g_co2e_per_kwh"].mean()
                ),
            }
        )
    return sorted(summaries, key=lambda row: (row["scenario"], row["mean_scenario_regret"]))


def select_champion_model(
    price_metrics_path: str | Path,
    carbon_metrics_path: str | Path,
    ranking_specific_metrics: list[dict[str, Any]],
    methodology: str,
) -> dict[str, Any]:
    """Select a champion model with carbon-first clean-hour scheduling weights."""
    price_metrics = load_json(price_metrics_path)
    carbon_metrics = load_json(carbon_metrics_path)
    price_summary = {
        row["model"]: row
        for row in price_metrics.get("summary", [])
    }
    carbon_summary = {
        row["model"]: row
        for row in carbon_metrics.get("summary", [])
        if row.get("methodology") == methodology
    }
    ranking_summary = {row["model"]: row for row in ranking_specific_metrics}
    models = sorted(set(price_summary) & set(carbon_summary) & set(ranking_summary))

    rows: list[dict[str, Any]] = []
    for model in models:
        top_5_ranking_loss = np.mean(
            [
                ranking_summary[model]["pairwise_ranking_loss"],
                1 - ranking_summary[model]["top_5_f1"],
            ]
        )
        rows.append(
            {
                "model": model,
                "price_direction_error": ranking_summary[model]["price_direction_error"],
                "carbon_intensity_mae_g_co2e_per_kwh": carbon_summary[model][
                    "carbon_intensity_mae_g_co2e_per_kwh"
                ],
                "carbon_regret_g_co2e_per_kwh": ranking_summary[model][
                    "mean_top_1_carbon_regret_g_co2e_per_kwh"
                ],
                "top_5_ranking_loss": float(top_5_ranking_loss),
                "pairwise_ranking_loss": ranking_summary[model]["pairwise_ranking_loss"],
                "top_5_f1": ranking_summary[model]["top_5_f1"],
                "mean_top_1_combined_regret": ranking_summary[model][
                    "mean_top_1_combined_regret"
                ],
                "price_mae_eur_mwh_reference_only": price_summary[model]["mae"],
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "champion_model": None,
            "weights": champion_weights(),
            "reason": "No overlapping model metrics were available.",
            "models": [],
        }

    frame["carbon_error_component"] = min_max_scale(
        frame["carbon_intensity_mae_g_co2e_per_kwh"]
    )
    frame["carbon_regret_component"] = min_max_scale(frame["carbon_regret_g_co2e_per_kwh"])
    frame["ranking_component"] = min_max_scale(frame["top_5_ranking_loss"])
    frame["price_direction_error_component"] = min_max_scale(
        frame["price_direction_error"]
    )
    weights = champion_weights()
    frame["champion_score"] = (
        weights["carbon_intensity_error"] * frame["carbon_error_component"]
        + weights["carbon_regret"] * frame["carbon_regret_component"]
        + weights["top_5_ranking_loss"] * frame["ranking_component"]
        + weights["price_direction_error"] * frame["price_direction_error_component"]
    )
    frame = frame.sort_values(["champion_score", "carbon_error_component"]).reset_index(drop=True)
    return {
        "champion_model": str(frame.iloc[0]["model"]),
        "weights": weights,
        "methodology": methodology,
        "selection_rule": (
            "Lowest weighted score wins. Carbon intensity error and carbon regret drive "
            "the clean-hour objective; price is used only as direction error versus the "
            "previous day at the same time."
        ),
        "models": frame.round(6).to_dict(orient="records"),
    }


def champion_weights() -> dict[str, float]:
    """Return champion-model selection weights."""
    return {
        "carbon_intensity_error": 0.45,
        "carbon_regret": 0.25,
        "top_5_ranking_loss": 0.2,
        "price_direction_error": 0.1,
    }


def min_max_scale(values: pd.Series) -> pd.Series:
    """Scale a lower-is-better series to [0, 1]."""
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or max_value == min_value:
        return pd.Series(0.0, index=values.index)
    return (values - min_value) / (max_value - min_value)


def safe_mean(values: pd.Series) -> float:
    """Return mean as float, preserving NaN when all values are missing."""
    value = values.astype(float).mean()
    return float(value) if not pd.isna(value) else float("nan")


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


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
