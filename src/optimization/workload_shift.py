"""Decision rankings for carbon- and cost-aware workload shifting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TIMESTAMP_COLUMN = "timestamp_utc"
ROOT = Path(__file__).resolve().parents[2]
RANKING_MODEL_PATH = ROOT / "models/ranking_top5_classifier.joblib"
RANKING_MODEL_METRICS_PATH = ROOT / "reports/metrics/ranking_model_metrics.json"
CONFIDENCE_CALIBRATION_PATH = ROOT / "reports/metrics/recommendation_confidence_calibration.json"
SCENARIO_CONFIDENCE_CALIBRATION_PATH = (
    ROOT / "reports/metrics/scenario_recommendation_confidence_calibration.json"
)
RECOMMENDATION_DRIFT_METRICS_PATH = ROOT / "reports/metrics/recommendation_drift_metrics.json"
PREDICTION_INTERVAL_CALIBRATION_PATH = (
    ROOT / "reports/metrics/recommendation_prediction_interval_calibration.json"
)
POLICY_BACKTEST_METRICS_PATH = ROOT / "reports/metrics/recommendation_policy_backtest.json"
SCENARIO_CHAMPION_SELECTION_PATH = ROOT / "reports/metrics/scenario_champion_selection.json"
UNCERTAINTY_GUARD_THRESHOLD = 0.85
UNCERTAINTY_GUARD_PENALTY = 0.15
MIN_CALIBRATION_BIN_ROWS = 30
MIN_CALIBRATION_GROUP_ROWS = 30
PREDICTION_INTERVAL_QUANTILE = 0.90
RECOMMENDATION_STATUS_OK = "recommended"
RECOMMENDATION_STATUS_NO_LOW_RISK = "no_low_risk_recommendation_available"
RANKING_MODEL_FEATURES = [
    "predicted_avg_price_eur_mwh",
    "predicted_avg_carbon_intensity_g_co2e_per_kwh",
    "predicted_total_emissions_kg_co2e",
    "predicted_price_rank",
    "predicted_carbon_rank",
    "candidate_count",
    "predicted_price_rank_pct",
    "predicted_carbon_rank_pct",
    "predicted_combined_score",
    "decision_uncertainty_score",
    "predicted_price_change_vs_previous_day_eur_mwh",
    "hour",
    "day_of_week",
]
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
    interval_calibration = build_prediction_interval_calibration(rankings)
    rankings = apply_prediction_interval_uncertainty(rankings, interval_calibration)
    rankings, ranking_model_report = apply_ranking_model_overlay(rankings)
    recommendations = build_top_workload_recommendations(rankings, top_n=top_n_recommendations)
    recommendations = add_recommendation_confidence(recommendations, rankings, top_n_recommendations)
    calibration = build_confidence_calibration(recommendations, top_n=top_n_recommendations)
    recommendations = apply_confidence_calibration(recommendations, calibration)
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
    scenario_calibration = build_confidence_calibration(
        scenario_recommendations,
        top_n=top_n_recommendations,
        group_column="scenario",
    )
    scenario_recommendations = apply_confidence_calibration(
        scenario_recommendations,
        scenario_calibration,
    )
    policy_backtest = summarize_policy_backtest(recommendations, scenario_recommendations)
    scenario_champions = select_scenario_champions(scenario_metrics["summary"])
    drift_metrics = summarize_recommendation_drift(recommendations, scenario_recommendations)
    champion_recommendations = build_champion_workload_recommendations(
        recommendations,
        champion["champion_model"],
    )
    validate_recommendation_artifacts(
        rankings=rankings,
        recommendations=recommendations,
        scenario_recommendations=scenario_recommendations,
        calibration=calibration,
        scenario_calibration=scenario_calibration,
        champion=champion,
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
    write_json(RANKING_MODEL_METRICS_PATH, ranking_model_report)
    write_json(CONFIDENCE_CALIBRATION_PATH, calibration)
    write_json(SCENARIO_CONFIDENCE_CALIBRATION_PATH, scenario_calibration)
    write_json(PREDICTION_INTERVAL_CALIBRATION_PATH, interval_calibration)
    write_json(POLICY_BACKTEST_METRICS_PATH, policy_backtest)
    write_json(SCENARIO_CHAMPION_SELECTION_PATH, scenario_champions)
    write_json(RECOMMENDATION_DRIFT_METRICS_PATH, drift_metrics)
    return {
        "constraints": asdict(constraints),
        "summary": metrics,
        "champion_model": champion["champion_model"],
        "ranking_model": ranking_model_report,
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
    candidates["base_predicted_combined_score"] = (
        price_weight * candidates["predicted_price_rank_pct"]
        + carbon_weight * candidates["predicted_carbon_rank_pct"]
    )
    candidates["decision_uncertainty_score"] = calculate_weighted_decision_uncertainty_scores(
        candidates,
        group_columns,
        price_weight,
        carbon_weight,
    )
    candidates["uncertainty_guard_penalty"] = calculate_uncertainty_guard_penalty(
        candidates["decision_uncertainty_score"],
    )
    candidates["predicted_combined_score"] = (
        candidates["base_predicted_combined_score"] + candidates["uncertainty_guard_penalty"]
    )
    candidates["uncertainty_guard_applied"] = candidates["uncertainty_guard_penalty"] > 0
    candidates["is_low_uncertainty_candidate"] = (
        candidates["decision_uncertainty_score"] <= UNCERTAINTY_GUARD_THRESHOLD
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
    candidates["baseline_predicted_combined_score"] = candidates["predicted_combined_score"]
    candidates["baseline_predicted_decision_rank"] = candidates["predicted_decision_rank"]
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


def apply_ranking_model_overlay(
    rankings: pd.DataFrame,
    model_path: str | Path = RANKING_MODEL_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train an out-of-window top-5 ranker and use it to rerank candidates."""
    frame = rankings.copy()
    frame["ranking_model_target_top_5"] = (frame["actual_decision_rank"] <= 5).astype(int)
    frame["ranking_model_score"] = np.nan
    frame["ranking_model_score_source"] = "fallback_baseline_score"
    model = build_ranking_model()
    feature_frame = build_ranking_feature_frame(frame)
    scored_rows = 0

    for model_name, model_frame in frame.groupby("model", observed=True):
        model_indices = model_frame.index
        for window in model_frame["window"].dropna().unique():
            validation_index = model_frame[model_frame["window"] == window].index
            train_index = model_indices.difference(validation_index)
            train_target = frame.loc[train_index, "ranking_model_target_top_5"]
            if len(train_index) < 50 or train_target.nunique() < 2:
                continue
            fitted = clone(model).fit(feature_frame.loc[train_index], train_target)
            frame.loc[validation_index, "ranking_model_score"] = fitted.predict_proba(
                feature_frame.loc[validation_index]
            )[:, 1]
            frame.loc[validation_index, "ranking_model_score_source"] = "out_of_window_classifier"
            scored_rows += len(validation_index)

    frame["ranking_model_score"] = frame["ranking_model_score"].fillna(
        1 - frame["predicted_combined_score"].clip(lower=0, upper=1)
    )
    frame["ranking_model_decision_score"] = 1 - frame["ranking_model_score"]
    classifier_scored = frame["ranking_model_score_source"] == "out_of_window_classifier"
    frame.loc[classifier_scored, "ranking_model_decision_score"] += frame.loc[
        classifier_scored,
        "uncertainty_guard_penalty",
    ]
    frame["predicted_decision_rank"] = rank_within_group(
        frame,
        ["window", "model", "decision_group"],
        "ranking_model_decision_score",
    )
    refresh_predicted_rank_flags(frame)
    report = summarize_ranking_model_overlay(frame, scored_rows)
    report["accepted_for_recommendations"] = ranking_model_improves_objective(report)
    report["acceptance_rule"] = (
        "Apply the learned ranker only when out-of-window combined regret and "
        "carbon regret are both no worse than the baseline rank score."
    )
    if report["accepted_for_recommendations"]:
        target = frame["ranking_model_target_top_5"]
        if len(frame) >= 50 and target.nunique() >= 2:
            fitted_final = clone(model).fit(feature_frame, target)
            write_ranking_model_artifact(
                model_path,
                {
                    "model_name": "global_top5_ranker",
                    "model": fitted_final,
                    "feature_columns": RANKING_MODEL_FEATURES,
                    "accepted_for_recommendations": True,
                    "trained_at_utc": datetime.now(UTC).isoformat(),
                },
            )
    else:
        remove_ranking_model_artifact(model_path)
        frame["predicted_decision_rank"] = frame["baseline_predicted_decision_rank"]
        frame["ranking_model_score_source"] = "guarded_fallback_baseline_score"
        refresh_predicted_rank_flags(frame)

    return frame.sort_values(
        ["window", "model", "decision_group", "predicted_decision_rank", TIMESTAMP_COLUMN]
    ).reset_index(drop=True), report


def apply_saved_ranking_model_overlay(
    rankings: pd.DataFrame,
    model_path: str | Path = RANKING_MODEL_PATH,
) -> pd.DataFrame:
    """Apply the persisted ranking model to operational candidate rankings."""
    frame = rankings.copy()
    artifact = load_ranking_model_artifact(model_path)
    if artifact is None or not artifact.get("accepted_for_recommendations", False):
        frame["ranking_model_score"] = 1 - frame["predicted_combined_score"].clip(lower=0, upper=1)
        frame["ranking_model_score_source"] = "fallback_baseline_score"
    else:
        feature_columns = artifact.get("feature_columns", RANKING_MODEL_FEATURES)
        features = build_ranking_feature_frame(frame, feature_columns)
        frame["ranking_model_score"] = artifact["model"].predict_proba(features)[:, 1]
        frame["ranking_model_score_source"] = "saved_classifier"
    frame["ranking_model_decision_score"] = 1 - frame["ranking_model_score"]
    classifier_scored = frame["ranking_model_score_source"] == "saved_classifier"
    frame.loc[classifier_scored, "ranking_model_decision_score"] += frame.loc[
        classifier_scored,
        "uncertainty_guard_penalty",
    ]
    frame["predicted_decision_rank"] = rank_within_group(
        frame,
        ["window", "model", "decision_group"],
        "ranking_model_decision_score",
    )
    refresh_predicted_rank_flags(frame)
    return frame.sort_values(
        ["window", "model", "decision_group", "predicted_decision_rank", TIMESTAMP_COLUMN]
    ).reset_index(drop=True)


def build_ranking_model() -> Pipeline:
    """Build the ranking-specific classifier used for candidate reranking."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=120,
                    max_leaf_nodes=15,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_ranking_feature_frame(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = RANKING_MODEL_FEATURES,
) -> pd.DataFrame:
    """Return model-ready ranking features."""
    output = frame.copy()
    timestamps = pd.to_datetime(output[TIMESTAMP_COLUMN], utc=True)
    output["hour"] = timestamps.dt.hour
    output["day_of_week"] = timestamps.dt.dayofweek
    for column in feature_columns:
        if column not in output:
            output[column] = np.nan
    return output[list(feature_columns)].apply(pd.to_numeric, errors="coerce")


def summarize_ranking_model_overlay(frame: pd.DataFrame, scored_rows: int) -> dict[str, Any]:
    """Summarize ranking-model quality against the base score."""
    predicted_best = frame[frame["predicted_decision_rank"] == 1]
    baseline_best = frame[frame["baseline_predicted_decision_rank"] == 1]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_type": "hist_gradient_boosting_top5_classifier",
        "feature_columns": RANKING_MODEL_FEATURES,
        "out_of_window_scored_rows": int(scored_rows),
        "rows": int(len(frame)),
        "learned_top_1_hit_rate": safe_float_mean(predicted_best["is_actual_best"]),
        "baseline_top_1_hit_rate": safe_float_mean(baseline_best["is_actual_best"]),
        "learned_mean_combined_regret": safe_float_mean(predicted_best["combined_regret"]),
        "baseline_mean_combined_regret": safe_float_mean(baseline_best["combined_regret"]),
        "learned_mean_carbon_regret_g_co2e_per_kwh": safe_float_mean(
            predicted_best["carbon_regret_g_co2e_per_kwh"]
        ),
        "baseline_mean_carbon_regret_g_co2e_per_kwh": safe_float_mean(
            baseline_best["carbon_regret_g_co2e_per_kwh"]
        ),
    }


def ranking_model_improves_objective(report: dict[str, Any]) -> bool:
    """Return whether the learned ranker clears the carbon-aware acceptance gate."""
    return (
        report["learned_mean_combined_regret"] <= report["baseline_mean_combined_regret"]
        and report["learned_mean_carbon_regret_g_co2e_per_kwh"]
        <= report["baseline_mean_carbon_regret_g_co2e_per_kwh"]
    )


def refresh_predicted_rank_flags(frame: pd.DataFrame) -> None:
    """Refresh boolean rank annotations after learned reranking."""
    frame["is_predicted_best"] = frame["predicted_decision_rank"] == 1
    frame["is_predicted_top_3"] = frame["predicted_decision_rank"] <= 3


def calculate_decision_uncertainty_scores(
    frame: pd.DataFrame,
    group_columns: list[str],
    score_column: str,
) -> pd.Series:
    """Estimate ranking uncertainty from score separation to adjacent candidates."""
    scored = frame[group_columns + [TIMESTAMP_COLUMN, score_column]].copy()
    scored = scored.sort_values(group_columns + [score_column, TIMESTAMP_COLUMN])
    scored["previous_score"] = scored.groupby(group_columns, observed=True)[score_column].shift(1)
    scored["next_score"] = scored.groupby(group_columns, observed=True)[score_column].shift(-1)
    previous_margin = scored[score_column] - scored["previous_score"]
    next_margin = scored["next_score"] - scored[score_column]
    nearest_margin = pd.concat([previous_margin, next_margin], axis=1).min(axis=1, skipna=True)
    score_range = scored.groupby(group_columns, observed=True)[score_column].transform(
        lambda values: values.max() - values.min()
    )
    uncertainty = 1 - (nearest_margin / score_range.replace(0, np.nan))
    uncertainty = uncertainty.fillna(1.0).clip(lower=0, upper=1)
    return uncertainty.reindex(frame.index)


def calculate_weighted_decision_uncertainty_scores(
    frame: pd.DataFrame,
    group_columns: list[str],
    price_weight: float,
    carbon_weight: float,
) -> pd.Series:
    """Estimate uncertainty from raw price and carbon forecast separation."""
    price_uncertainty = calculate_decision_uncertainty_scores(
        frame,
        group_columns,
        "predicted_avg_price_eur_mwh",
    )
    carbon_uncertainty = calculate_decision_uncertainty_scores(
        frame,
        group_columns,
        "predicted_avg_carbon_intensity_g_co2e_per_kwh",
    )
    weight_sum = price_weight + carbon_weight
    if weight_sum == 0:
        return pd.Series(0.0, index=frame.index)
    return (
        (price_weight * price_uncertainty + carbon_weight * carbon_uncertainty) / weight_sum
    ).clip(lower=0, upper=1)


def calculate_uncertainty_guard_penalty(
    uncertainty: pd.Series,
    threshold: float = UNCERTAINTY_GUARD_THRESHOLD,
    max_penalty: float = UNCERTAINTY_GUARD_PENALTY,
) -> pd.Series:
    """Return a lower-is-better score penalty for high-uncertainty candidates."""
    if threshold >= 1:
        return pd.Series(0.0, index=uncertainty.index)
    excess = ((uncertainty - threshold) / (1 - threshold)).clip(lower=0, upper=1)
    return excess * max_penalty


def build_prediction_interval_calibration(
    rankings: pd.DataFrame,
    quantile: float = PREDICTION_INTERVAL_QUANTILE,
) -> dict[str, Any]:
    """Build empirical prediction interval widths from historical candidate residuals."""
    if rankings.empty:
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "quantile": quantile,
            "models": {},
            "fallback": default_prediction_interval_widths(),
        }
    frame = rankings.copy()
    frame["price_abs_error"] = (
        frame["predicted_avg_price_eur_mwh"] - frame["actual_avg_price_eur_mwh"]
    ).abs()
    frame["carbon_abs_error"] = (
        frame["predicted_avg_carbon_intensity_g_co2e_per_kwh"]
        - frame["actual_avg_carbon_intensity_g_co2e_per_kwh"]
    ).abs()
    model_widths = {
        str(model): prediction_interval_widths(model_frame, quantile)
        for model, model_frame in frame.groupby("model", observed=True)
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "Empirical candidate-level absolute residual quantiles.",
        "quantile": quantile,
        "models": model_widths,
        "fallback": prediction_interval_widths(frame, quantile),
    }


def prediction_interval_widths(frame: pd.DataFrame, quantile: float) -> dict[str, float]:
    """Return JSON-safe empirical interval half-widths for one slice."""
    return {
        "price_interval_half_width_eur_mwh": safe_quantile(
            frame["price_abs_error"],
            quantile,
        ),
        "carbon_interval_half_width_g_co2e_per_kwh": safe_quantile(
            frame["carbon_abs_error"],
            quantile,
        ),
        "rows": int(len(frame)),
    }


def default_prediction_interval_widths() -> dict[str, float]:
    """Return conservative zero-width fallback interval metadata."""
    return {
        "price_interval_half_width_eur_mwh": 0.0,
        "carbon_interval_half_width_g_co2e_per_kwh": 0.0,
        "rows": 0,
    }


def apply_prediction_interval_uncertainty(
    rankings: pd.DataFrame,
    calibration: dict[str, Any] | None,
) -> pd.DataFrame:
    """Add calibrated interval widths and rerank with interval-aware uncertainty."""
    frame = rankings.copy()
    if frame.empty:
        return frame
    interval_rows = [prediction_interval_row(str(model), calibration) for model in frame["model"]]
    frame["predicted_price_interval_half_width_eur_mwh"] = [
        row["price_interval_half_width_eur_mwh"] for row in interval_rows
    ]
    frame["predicted_carbon_interval_half_width_g_co2e_per_kwh"] = [
        row["carbon_interval_half_width_g_co2e_per_kwh"] for row in interval_rows
    ]
    group_columns = ["window", "model", "decision_group"]
    price_range = frame.groupby(group_columns, observed=True)[
        "predicted_avg_price_eur_mwh"
    ].transform(lambda values: values.max() - values.min())
    carbon_range = frame.groupby(group_columns, observed=True)[
        "predicted_avg_carbon_intensity_g_co2e_per_kwh"
    ].transform(lambda values: values.max() - values.min())
    price_interval_score = (
        frame["predicted_price_interval_half_width_eur_mwh"] / price_range.replace(0, np.nan)
    )
    carbon_interval_score = (
        frame["predicted_carbon_interval_half_width_g_co2e_per_kwh"]
        / carbon_range.replace(0, np.nan)
    )
    frame["prediction_interval_uncertainty_score"] = pd.concat(
        [price_interval_score, carbon_interval_score],
        axis=1,
    ).max(axis=1, skipna=True).fillna(0.0).clip(lower=0, upper=1)
    frame["decision_uncertainty_score"] = pd.concat(
        [frame["decision_uncertainty_score"], frame["prediction_interval_uncertainty_score"]],
        axis=1,
    ).max(axis=1, skipna=True).clip(lower=0, upper=1)
    frame["uncertainty_guard_penalty"] = calculate_uncertainty_guard_penalty(
        frame["decision_uncertainty_score"]
    )
    frame["predicted_combined_score"] = (
        frame["base_predicted_combined_score"] + frame["uncertainty_guard_penalty"]
    )
    frame["uncertainty_guard_applied"] = frame["uncertainty_guard_penalty"] > 0
    frame["is_low_uncertainty_candidate"] = (
        frame["decision_uncertainty_score"] <= UNCERTAINTY_GUARD_THRESHOLD
    )
    frame["predicted_decision_rank"] = rank_within_group(
        frame,
        group_columns,
        "predicted_combined_score",
    )
    frame["baseline_predicted_combined_score"] = frame["predicted_combined_score"]
    frame["baseline_predicted_decision_rank"] = frame["predicted_decision_rank"]
    refresh_predicted_rank_flags(frame)
    return frame.sort_values(
        group_columns + ["predicted_decision_rank", TIMESTAMP_COLUMN]
    ).reset_index(drop=True)


def prediction_interval_row(model: str, calibration: dict[str, Any] | None) -> dict[str, float]:
    """Return prediction interval widths for a model."""
    if not calibration:
        return default_prediction_interval_widths()
    return calibration.get("models", {}).get(
        model,
        calibration.get("fallback", default_prediction_interval_widths()),
    )


def write_ranking_model_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    """Persist a ranking model artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)


def remove_ranking_model_artifact(path: str | Path) -> None:
    """Remove a stale accepted ranking model artifact when the gate fails."""
    artifact_path = Path(path)
    if artifact_path.exists():
        artifact_path.unlink()


def load_ranking_model_artifact(path: str | Path) -> dict[str, Any] | None:
    """Load a persisted ranking model artifact if available."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)


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
                "mean_decision_uncertainty_score": safe_float_mean(
                    predicted_best["decision_uncertainty_score"]
                ),
                "high_uncertainty_recommendation_share": safe_float_mean(
                    predicted_best["decision_uncertainty_score"] > UNCERTAINTY_GUARD_THRESHOLD
                ),
                "uncertainty_guard_applied_share": safe_float_mean(
                    predicted_best["uncertainty_guard_applied"]
                ),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_combined_regret"])


def build_top_workload_recommendations(rankings: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top-N recommended workload start times per model/window/decision group."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    group_columns = ["window", "model", "decision_group"]
    recommended_frames: list[pd.DataFrame] = []
    for _, group in rankings.sort_values(
        group_columns + ["predicted_decision_rank", TIMESTAMP_COLUMN]
    ).groupby(group_columns, observed=True):
        low_risk = group[group["is_low_uncertainty_candidate"]]
        if low_risk.empty:
            selected = group.head(1).copy()
            selected["recommendation_status"] = RECOMMENDATION_STATUS_NO_LOW_RISK
            selected["suppressed_by_uncertainty_guard"] = True
        else:
            selected = low_risk.head(top_n).copy()
            selected["recommendation_status"] = RECOMMENDATION_STATUS_OK
            selected["suppressed_by_uncertainty_guard"] = False
        selected["recommendation_rank"] = range(1, len(selected) + 1)
        selected["eligible_low_uncertainty_candidate_count"] = int(len(low_risk))
        recommended_frames.append(selected)

    recommended = pd.concat(recommended_frames, ignore_index=True) if recommended_frames else rankings.iloc[0:0].copy()
    recommendation_columns = [
        "window",
        "model",
        "decision_group",
        "recommendation_rank",
        "recommendation_status",
        "suppressed_by_uncertainty_guard",
        "eligible_low_uncertainty_candidate_count",
        TIMESTAMP_COLUMN,
        "workload_end_utc",
        "duration_hours",
        "predicted_combined_score",
        "baseline_predicted_decision_rank",
        "ranking_model_score",
        "ranking_model_decision_score",
        "ranking_model_score_source",
        "decision_uncertainty_score",
        "prediction_interval_uncertainty_score",
        "predicted_price_interval_half_width_eur_mwh",
        "predicted_carbon_interval_half_width_g_co2e_per_kwh",
        "uncertainty_guard_penalty",
        "uncertainty_guard_applied",
        "is_low_uncertainty_candidate",
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
    for column in recommendation_columns:
        if column not in recommended:
            recommended[column] = np.nan
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


def build_confidence_calibration(
    recommendations: pd.DataFrame,
    top_n: int = 5,
    group_column: str | None = None,
    min_bin_rows: int = MIN_CALIBRATION_BIN_ROWS,
    min_group_rows: int = MIN_CALIBRATION_GROUP_ROWS,
) -> dict[str, Any]:
    """Build empirical confidence calibration from historical recommendations."""
    if recommendations.empty or "actual_decision_rank" not in recommendations:
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "top_n": top_n,
            "group_column": group_column,
            "min_bin_rows": min_bin_rows,
            "min_group_rows": min_group_rows,
            "bins": [],
            "groups": {},
            "group_fallbacks": {},
            "fallback": default_confidence_fallback(),
        }
    frame = recommendations.copy()
    frame["actual_top_n"] = frame["actual_decision_rank"] <= top_n
    frame["actual_top_1"] = frame["actual_decision_rank"] == 1
    group_bins = (
        {
            str(group): build_confidence_bins(group_frame, min_bin_rows=min_bin_rows)
            for group, group_frame in frame.groupby(group_column, observed=True)
            if len(group_frame) >= min_group_rows
        }
        if group_column and group_column in frame
        else {}
    )
    group_fallbacks = (
        {
            str(group): default_confidence_fallback(group_frame)
            for group, group_frame in frame.groupby(group_column, observed=True)
            if len(group_frame) >= min_group_rows
        }
        if group_column and group_column in frame
        else {}
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "Confidence is calibrated by mapping heuristic confidence bins to "
            "historical top-N hit rates and observed regret."
        ),
        "top_n": top_n,
        "group_column": group_column,
        "min_bin_rows": min_bin_rows,
        "min_group_rows": min_group_rows,
        "bins": build_confidence_bins(frame, min_bin_rows=min_bin_rows),
        "groups": group_bins,
        "group_fallbacks": group_fallbacks,
        "fallback": default_confidence_fallback(frame),
    }


def build_confidence_bins(
    frame: pd.DataFrame,
    min_bin_rows: int = MIN_CALIBRATION_BIN_ROWS,
) -> list[dict[str, Any]]:
    """Build empirical calibration bins for one recommendation slice."""
    bins: list[dict[str, Any]] = []
    for label, low, high in [
        ("low", 0.0, 0.5),
        ("medium", 0.5, 0.75),
        ("high", 0.75, 1.01),
    ]:
        bin_frame = frame[(frame["confidence_score"] >= low) & (frame["confidence_score"] < high)]
        if len(bin_frame) < min_bin_rows:
            continue
        bins.append(
            {
                "bin": label,
                "min_score": low,
                "max_score": high,
                "rows": int(len(bin_frame)),
                "empirical_top_n_hit_rate": safe_float_mean(bin_frame["actual_top_n"]),
                "empirical_top_1_hit_rate": safe_float_mean(bin_frame["actual_top_1"]),
                "mean_combined_regret": safe_float_mean(bin_frame["combined_regret"]),
                "mean_carbon_regret_g_co2e_per_kwh": safe_float_mean(
                    bin_frame["carbon_regret_g_co2e_per_kwh"]
                ),
                "mean_cost_regret_eur_mwh": safe_float_mean(bin_frame["cost_regret_eur_mwh"]),
            }
        )
    return bins


def apply_confidence_calibration(
    recommendations: pd.DataFrame,
    calibration: dict[str, Any] | None,
) -> pd.DataFrame:
    """Apply empirical confidence calibration to recommendation rows."""
    output = recommendations.copy()
    if output.empty:
        return output
    output["heuristic_confidence_score"] = output["confidence_score"]
    output["heuristic_confidence_level"] = output["confidence_level"]
    rows = [
        calibration_row_for_score(float(row["heuristic_confidence_score"]), calibration, row)
        for row in output.to_dict(orient="records")
    ]
    output["empirical_top_n_hit_rate"] = [row["empirical_top_n_hit_rate"] for row in rows]
    output["empirical_top_1_hit_rate"] = [row["empirical_top_1_hit_rate"] for row in rows]
    output["expected_combined_regret"] = [row["mean_combined_regret"] for row in rows]
    output["expected_carbon_regret_g_co2e_per_kwh"] = [
        row["mean_carbon_regret_g_co2e_per_kwh"] for row in rows
    ]
    output["expected_cost_regret_eur_mwh"] = [row["mean_cost_regret_eur_mwh"] for row in rows]
    output["confidence_score"] = (
        0.55 * output["heuristic_confidence_score"]
        + 0.45 * output["empirical_top_n_hit_rate"]
    ).clip(lower=0, upper=1)
    output["confidence_level"] = output["confidence_score"].map(confidence_label)
    float_columns = output.select_dtypes(include=["float"]).columns
    output[float_columns] = output[float_columns].round(2)
    return output


def calibration_row_for_score(
    score: float,
    calibration: dict[str, Any] | None,
    recommendation_row: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return the calibration bin that contains a heuristic score."""
    if calibration:
        group_column = calibration.get("group_column")
        group_value = (
            str(recommendation_row.get(group_column))
            if group_column and recommendation_row and recommendation_row.get(group_column) is not None
            else None
        )
        bins = calibration.get("groups", {}).get(group_value, calibration.get("bins", []))
        for row in bins:
            if score >= row["min_score"] and score < row["max_score"]:
                return row
        fallback = calibration.get("group_fallbacks", {}).get(group_value)
        if fallback:
            return fallback
    fallback = (
        calibration.get("fallback", default_confidence_fallback())
        if calibration
        else default_confidence_fallback()
    )
    return fallback


def load_confidence_calibration(
    path: str | Path = CONFIDENCE_CALIBRATION_PATH,
) -> dict[str, Any] | None:
    """Load confidence calibration metadata if available."""
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    return load_json(calibration_path)


def validate_recommendation_artifacts(
    rankings: pd.DataFrame,
    recommendations: pd.DataFrame,
    scenario_recommendations: pd.DataFrame,
    calibration: dict[str, Any],
    scenario_calibration: dict[str, Any],
    champion: dict[str, Any],
) -> None:
    """Validate core recommendation artifact schemas before writing them."""
    validate_columns(
        rankings,
        [
            TIMESTAMP_COLUMN,
            "window",
            "model",
            "decision_group",
            "predicted_decision_rank",
            "actual_decision_rank",
            "combined_regret",
            "decision_uncertainty_score",
            "uncertainty_guard_applied",
        ],
        "workload rankings",
    )
    validate_columns(
        recommendations,
        [
            "window",
            "model",
            "decision_group",
            "recommendation_rank",
            "recommendation_status",
            TIMESTAMP_COLUMN,
            "confidence_score",
            "confidence_level",
            "expected_combined_regret",
            "decision_uncertainty_score",
            "prediction_interval_uncertainty_score",
        ],
        "top-5 recommendations",
    )
    validate_columns(
        scenario_recommendations,
        [
            "scenario",
            "window",
            "model",
            "decision_group",
            "recommendation_rank",
            "recommendation_status",
            TIMESTAMP_COLUMN,
            "confidence_score",
            "confidence_level",
            "expected_combined_regret",
            "decision_uncertainty_score",
        ],
        "scenario recommendations",
    )
    validate_mapping_keys(
        calibration,
        ["generated_at_utc", "top_n", "bins", "fallback"],
        "confidence calibration",
    )
    validate_mapping_keys(
        scenario_calibration,
        ["generated_at_utc", "top_n", "group_column", "bins", "groups", "fallback"],
        "scenario confidence calibration",
    )
    validate_mapping_keys(
        champion,
        ["champion_model", "weights", "models"],
        "champion selection",
    )


def validate_columns(frame: pd.DataFrame, required_columns: list[str], artifact: str) -> None:
    """Raise a clear error when an artifact frame is missing required columns."""
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {missing}")


def validate_mapping_keys(payload: dict[str, Any], required_keys: list[str], artifact: str) -> None:
    """Raise a clear error when an artifact payload is missing required keys."""
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{artifact} missing required keys: {missing}")


def default_confidence_fallback(frame: pd.DataFrame | None = None) -> dict[str, float]:
    """Return fallback confidence calibration values."""
    if frame is None or frame.empty or "actual_decision_rank" not in frame:
        return {
            "empirical_top_n_hit_rate": 0.5,
            "empirical_top_1_hit_rate": 0.2,
            "mean_combined_regret": 0.0,
            "mean_carbon_regret_g_co2e_per_kwh": 0.0,
            "mean_cost_regret_eur_mwh": 0.0,
        }
    return {
        "empirical_top_n_hit_rate": safe_float_mean(frame["actual_decision_rank"] <= 5),
        "empirical_top_1_hit_rate": safe_float_mean(frame["actual_decision_rank"] == 1),
        "mean_combined_regret": safe_float_mean(frame["combined_regret"]),
        "mean_carbon_regret_g_co2e_per_kwh": safe_float_mean(
            frame["carbon_regret_g_co2e_per_kwh"]
        ),
        "mean_cost_regret_eur_mwh": safe_float_mean(frame["cost_regret_eur_mwh"]),
    }


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
        "by_day": summarize_ranking_metrics_by_day(by_decision_group),
    }


def summarize_ranking_metrics_by_day(
    by_decision_group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate ranking-specific evaluation by calendar day."""
    if not by_decision_group:
        return []
    frame = pd.DataFrame(by_decision_group)
    frame["decision_day"] = frame["decision_group"].astype(str).str.slice(0, 10)
    summaries: list[dict[str, Any]] = []
    for (window, model, decision_day), group in frame.groupby(
        ["window", "model", "decision_day"],
        observed=True,
    ):
        summaries.append(
            {
                "window": window,
                "model": model,
                "decision_day": decision_day,
                "decision_groups": int(len(group)),
                "pairwise_ranking_loss": safe_float_mean(group["pairwise_ranking_loss"]),
                "top_5_precision": safe_float_mean(group["top_5_precision"]),
                "top_5_recall": safe_float_mean(group["top_5_recall"]),
                "top_5_f1": safe_float_mean(group["top_5_f1"]),
                "mean_top_1_combined_regret": safe_float_mean(
                    group["top_1_combined_regret"]
                ),
                "mean_top_1_carbon_regret_g_co2e_per_kwh": safe_float_mean(
                    group["top_1_carbon_regret_g_co2e_per_kwh"]
                ),
                "mean_top_1_cost_regret_eur_mwh": safe_float_mean(
                    group["top_1_cost_regret_eur_mwh"]
                ),
            }
        )
    return summaries


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
        scenario_frame["actual_decision_rank"] = scenario_frame["actual_scenario_rank"]
        scenario_frame["decision_uncertainty_score"] = calculate_decision_uncertainty_scores(
            scenario_frame,
            scenario_group_columns,
            "predicted_scenario_score",
        )
        scenario_frame["is_low_uncertainty_candidate"] = (
            scenario_frame["decision_uncertainty_score"] <= UNCERTAINTY_GUARD_THRESHOLD
        )
        scenario_frame["actual_best_scenario_score"] = scenario_frame.groupby(
            scenario_group_columns,
            observed=True,
        )["actual_scenario_score"].transform("min")
        scenario_frame["scenario_regret"] = (
            scenario_frame["actual_scenario_score"]
            - scenario_frame["actual_best_scenario_score"]
        )
        scenario_frame["combined_regret"] = scenario_frame["scenario_regret"]
        scenario_frame["confidence_score"] = calculate_scenario_confidence_scores(
            scenario_frame,
            top_n=top_n,
        )
        scenario_frame["confidence_level"] = scenario_frame["confidence_score"].map(
            confidence_label
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
    group_columns = ["scenario", "window", "model", "decision_group"]
    recommended_frames: list[pd.DataFrame] = []
    for _, group in rankings.sort_values(
        group_columns + ["predicted_scenario_rank", TIMESTAMP_COLUMN]
    ).groupby(group_columns, observed=True):
        low_risk = group[group["is_low_uncertainty_candidate"]]
        if low_risk.empty:
            selected = group.head(1).copy()
            selected["recommendation_status"] = RECOMMENDATION_STATUS_NO_LOW_RISK
            selected["suppressed_by_uncertainty_guard"] = True
        else:
            selected = low_risk.head(top_n).copy()
            selected["recommendation_status"] = RECOMMENDATION_STATUS_OK
            selected["suppressed_by_uncertainty_guard"] = False
        selected["recommendation_rank"] = range(1, len(selected) + 1)
        selected["eligible_low_uncertainty_candidate_count"] = int(len(low_risk))
        recommended_frames.append(selected)

    recommended = (
        pd.concat(recommended_frames, ignore_index=True)
        if recommended_frames
        else rankings.iloc[0:0].copy()
    )
    columns = [
        "scenario",
        "scenario_price_weight",
        "scenario_carbon_weight",
        "window",
        "model",
        "decision_group",
        "recommendation_rank",
        "recommendation_status",
        "suppressed_by_uncertainty_guard",
        "eligible_low_uncertainty_candidate_count",
        TIMESTAMP_COLUMN,
        "workload_end_utc",
        "duration_hours",
        "predicted_scenario_score",
        "confidence_score",
        "confidence_level",
        "decision_uncertainty_score",
        "prediction_interval_uncertainty_score",
        "predicted_price_interval_half_width_eur_mwh",
        "predicted_carbon_interval_half_width_g_co2e_per_kwh",
        "is_low_uncertainty_candidate",
        "predicted_price_direction_vs_previous_day",
        "predicted_avg_carbon_intensity_g_co2e_per_kwh",
        "predicted_total_emissions_kg_co2e",
        "predicted_carbon_rank",
        "actual_scenario_rank",
        "actual_decision_rank",
        "scenario_regret",
        "combined_regret",
        "cost_regret_eur_mwh",
        "carbon_regret_g_co2e_per_kwh",
        "carbon_savings_vs_run_now_g_co2e_per_kwh",
    ]
    for column in columns:
        if column not in recommended:
            recommended[column] = np.nan
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
        top_5_metrics = calculate_scenario_top_k_metrics(model_frame, top_k=5)
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
                **top_5_metrics,
                "pairwise_ranking_loss": safe_float_mean(
                    model_frame.groupby(group_columns, observed=True).apply(
                        lambda group: calculate_pairwise_ranking_loss(
                            predicted=group["predicted_scenario_score"].to_numpy(dtype=float),
                            actual=group["actual_scenario_score"].to_numpy(dtype=float),
                        ),
                        include_groups=False,
                    )
                ),
                "mean_scenario_regret": float(predicted_best["scenario_regret"].mean()),
                "mean_carbon_regret_g_co2e_per_kwh": float(
                    predicted_best["carbon_regret_g_co2e_per_kwh"].mean()
                ),
                "mean_carbon_savings_vs_run_now_g_co2e_per_kwh": float(
                    predicted_best["carbon_savings_vs_run_now_g_co2e_per_kwh"].mean()
                ),
                "mean_decision_uncertainty_score": safe_float_mean(
                    predicted_best["decision_uncertainty_score"]
                ),
                "high_uncertainty_recommendation_share": safe_float_mean(
                    predicted_best["decision_uncertainty_score"] > UNCERTAINTY_GUARD_THRESHOLD
                ),
            }
        )
    return sorted(summaries, key=lambda row: (row["scenario"], row["mean_scenario_regret"]))


def calculate_scenario_confidence_scores(rankings: pd.DataFrame, top_n: int) -> pd.Series:
    """Calculate confidence scores for scenario-specific recommendation ranks."""
    group_columns = ["scenario", "window", "model", "decision_group"]
    frame = rankings.sort_values(group_columns + ["predicted_scenario_rank"]).copy()
    frame["next_score"] = frame.groupby(group_columns, observed=True)[
        "predicted_scenario_score"
    ].shift(-1)
    frame["score_range"] = frame.groupby(group_columns, observed=True)[
        "predicted_scenario_score"
    ].transform(lambda values: values.max() - values.min())
    margin = (
        (frame["next_score"] - frame["predicted_scenario_score"])
        / frame["score_range"].replace(0, np.nan)
    ).clip(lower=0, upper=1)
    rank_denominator = max(top_n - 1, 1)
    rank_component = 1 - ((frame["predicted_scenario_rank"] - 1) / rank_denominator)
    confidence = (
        0.55 * rank_component
        + 0.25 * margin.fillna(0)
        + 0.20 * (1 - frame["decision_uncertainty_score"])
    ).clip(lower=0, upper=1)
    return confidence.reindex(rankings.index)


def calculate_scenario_top_k_metrics(rankings: pd.DataFrame, top_k: int) -> dict[str, float]:
    """Evaluate top-k scenario recommendations across decision groups."""
    metrics = []
    group_columns = ["scenario", "window", "decision_group"]
    for _, group in rankings.groupby(group_columns, observed=True):
        predicted_top = group["predicted_scenario_rank"] <= top_k
        actual_top = group["actual_scenario_rank"] <= top_k
        true_positive = int((predicted_top & actual_top).sum())
        predicted_positive = int(predicted_top.sum())
        actual_positive = int(actual_top.sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append(
            {
                f"top_{top_k}_precision": float(precision),
                f"top_{top_k}_recall": float(recall),
                f"top_{top_k}_f1": float(f1),
            }
        )
    if not metrics:
        return {
            f"top_{top_k}_precision": 0.0,
            f"top_{top_k}_recall": 0.0,
            f"top_{top_k}_f1": 0.0,
        }
    metric_frame = pd.DataFrame(metrics)
    return {
        column: safe_float_mean(metric_frame[column])
        for column in metric_frame.columns
    }


def summarize_recommendation_drift(
    recommendations: pd.DataFrame,
    scenario_recommendations: pd.DataFrame | None = None,
    previous_recommendations: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize recommendation behavior for drift monitoring."""
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "recommendations": summarize_recommendation_distribution(recommendations),
    }
    if scenario_recommendations is not None:
        payload["scenario_recommendations"] = summarize_recommendation_distribution(
            scenario_recommendations
        )
    if previous_recommendations is not None and not previous_recommendations.empty:
        payload["rank_overlap_with_previous"] = calculate_rank_overlap(
            recommendations,
            previous_recommendations,
        )
    return payload


def summarize_recommendation_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact distribution metrics for recommendation drift checks."""
    if frame.empty:
        return {
            "rows": 0,
            "decision_groups": 0,
            "average_confidence_score": None,
            "high_confidence_share": None,
            "high_uncertainty_share": None,
            "mean_decision_uncertainty_score": None,
            "mean_predicted_carbon_intensity_g_co2e_per_kwh": None,
            "recommendation_status_counts": {},
        }
    status_counts = (
        frame["recommendation_status"].value_counts(dropna=False).to_dict()
        if "recommendation_status" in frame
        else {}
    )
    return {
        "rows": int(len(frame)),
        "decision_groups": int(frame["decision_group"].nunique())
        if "decision_group" in frame
        else 0,
        "average_confidence_score": safe_optional_float_mean(frame, "confidence_score"),
        "high_confidence_share": safe_optional_float_mean(
            frame["confidence_level"] == "high"
        )
        if "confidence_level" in frame
        else None,
        "high_uncertainty_share": safe_optional_float_mean(
            frame["decision_uncertainty_score"] > UNCERTAINTY_GUARD_THRESHOLD
        )
        if "decision_uncertainty_score" in frame
        else None,
        "mean_decision_uncertainty_score": safe_optional_float_mean(
            frame,
            "decision_uncertainty_score",
        ),
        "mean_predicted_carbon_intensity_g_co2e_per_kwh": safe_optional_float_mean(
            frame,
            "predicted_avg_carbon_intensity_g_co2e_per_kwh",
        ),
        "recommendation_status_counts": {
            str(status): int(count) for status, count in status_counts.items()
        },
    }


def calculate_rank_overlap(current: pd.DataFrame, previous: pd.DataFrame) -> float | None:
    """Return share of current top-ranked timestamps seen in previous output."""
    required = {"decision_group", TIMESTAMP_COLUMN}
    if not required.issubset(current.columns) or not required.issubset(previous.columns):
        return None
    current_keys = set(zip(current["decision_group"], current[TIMESTAMP_COLUMN], strict=False))
    previous_keys = set(zip(previous["decision_group"], previous[TIMESTAMP_COLUMN], strict=False))
    if not current_keys:
        return None
    return round(len(current_keys & previous_keys) / len(current_keys), 4)


def summarize_policy_backtest(
    recommendations: pd.DataFrame,
    scenario_recommendations: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate realized recommendation policy regret from emitted recommendation rows."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "Historical day-by-day backtest over the same top recommendation rows "
            "emitted by the recommendation engine."
        ),
        "base_policy": summarize_policy_backtest_slice(recommendations),
        "scenario_policy": summarize_policy_backtest_slice(
            scenario_recommendations,
            group_columns=["scenario", "model"],
        ),
    }


def summarize_policy_backtest_slice(
    recommendations: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Summarize rank-1 policy outcomes for one recommendation table."""
    if recommendations.empty:
        return []
    groups = group_columns or ["model"]
    rank_column = (
        "actual_scenario_rank" if "actual_scenario_rank" in recommendations else "actual_decision_rank"
    )
    top = recommendations[recommendations["recommendation_rank"] == 1].copy()
    summaries: list[dict[str, Any]] = []
    for group_key, group in top.groupby(groups, observed=True):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {
            column: value for column, value in zip(groups, group_key_values, strict=False)
        }
        row.update(
            {
                "decision_groups": int(group["decision_group"].nunique()),
                "recommended_groups": int(
                    (group["recommendation_status"] == RECOMMENDATION_STATUS_OK).sum()
                )
                if "recommendation_status" in group
                else int(len(group)),
                "no_low_risk_groups": int(
                    (group["recommendation_status"] == RECOMMENDATION_STATUS_NO_LOW_RISK).sum()
                )
                if "recommendation_status" in group
                else 0,
                "top_1_hit_rate": safe_optional_float_mean(group[rank_column] == 1),
                "top_5_hit_rate": safe_optional_float_mean(group[rank_column] <= 5),
                "mean_combined_regret": safe_optional_float_mean(group, "combined_regret"),
                "mean_carbon_regret_g_co2e_per_kwh": safe_optional_float_mean(
                    group,
                    "carbon_regret_g_co2e_per_kwh",
                ),
                "mean_cost_regret_eur_mwh": safe_optional_float_mean(
                    group,
                    "cost_regret_eur_mwh",
                ),
                "mean_confidence_score": safe_optional_float_mean(group, "confidence_score"),
                "mean_decision_uncertainty_score": safe_optional_float_mean(
                    group,
                    "decision_uncertainty_score",
                ),
            }
        )
        summaries.append(row)
    return sorted(summaries, key=lambda row: row.get("mean_combined_regret") or 0.0)


def select_scenario_champions(scenario_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest-regret model per scenario."""
    if not scenario_metrics:
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "selection_rule": "Lowest scenario regret wins per scenario.",
            "champions": [],
        }
    frame = pd.DataFrame(scenario_metrics).copy()
    frame["scenario_champion_score"] = (
        0.50 * min_max_scale(frame["mean_scenario_regret"])
        + 0.25 * min_max_scale(frame["mean_carbon_regret_g_co2e_per_kwh"])
        + 0.25 * min_max_scale(1 - frame["top_5_f1"])
    )
    champions = (
        frame.sort_values(["scenario", "scenario_champion_score", "mean_scenario_regret"])
        .groupby("scenario", observed=True)
        .head(1)
        .reset_index(drop=True)
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "Lowest scenario-specific weighted score wins: 50% scenario regret, "
            "25% carbon regret, 25% top-5 loss."
        ),
        "champions": champions.round(6).to_dict(orient="records"),
        "models": frame.round(6).to_dict(orient="records"),
    }


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
                "price_mae_eur_mwh_reference_only": price_summary[model].get("mae"),
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
    frame["combined_regret_component"] = min_max_scale(frame["mean_top_1_combined_regret"])
    frame["ranking_component"] = min_max_scale(frame["top_5_ranking_loss"])
    frame["price_direction_error_component"] = min_max_scale(
        frame["price_direction_error"]
    )
    weights = champion_weights()
    frame["champion_score"] = (
        weights["recommendation_regret"] * frame["combined_regret_component"]
        + weights["carbon_regret"] * frame["carbon_regret_component"]
        + weights["top_5_ranking_loss"] * frame["ranking_component"]
        + weights["price_direction_error"] * frame["price_direction_error_component"]
        + weights["carbon_intensity_error"] * frame["carbon_error_component"]
    )
    frame = frame.sort_values(["champion_score", "combined_regret_component"]).reset_index(
        drop=True
    )
    return {
        "champion_model": str(frame.iloc[0]["model"]),
        "weights": weights,
        "methodology": methodology,
        "selection_rule": (
            "Lowest weighted score wins. Realized recommendation regret is the primary "
            "objective; carbon forecast MAE is retained as a small reference term."
        ),
        "models": frame.round(6).to_dict(orient="records"),
    }


def champion_weights() -> dict[str, float]:
    """Return champion-model selection weights."""
    return {
        "recommendation_regret": 0.35,
        "carbon_regret": 0.25,
        "top_5_ranking_loss": 0.2,
        "price_direction_error": 0.1,
        "carbon_intensity_error": 0.1,
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


def safe_float_mean(values: pd.Series) -> float:
    """Return a JSON-safe mean, using zero when all values are missing."""
    value = values.astype(float).mean()
    return float(value) if not pd.isna(value) else 0.0


def safe_quantile(values: pd.Series, quantile: float) -> float:
    """Return a JSON-safe quantile for numeric values."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.quantile(quantile))


def safe_optional_float_mean(
    values: pd.DataFrame | pd.Series,
    column: str | None = None,
) -> float | None:
    """Return a rounded mean when values are available."""
    series = values[column] if column else values
    if len(series) == 0:
        return None
    value = series.astype(float).mean()
    return round(float(value), 4) if not pd.isna(value) else None


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
