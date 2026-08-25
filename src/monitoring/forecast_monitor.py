"""Forecast and recommendation monitoring for ingestion-only refreshes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.carbon.intensity import load_emission_factor_config
from src.data.pipeline_health import DEFAULT_OUTPUT_PATH as PIPELINE_HEALTH_PATH
from src.models.baseline_price import TIMESTAMP_COLUMN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "reports/metrics/forecast_monitoring.json"
FEATURES_PATH = ROOT / "data/processed/modeling_price_features.csv"
WORKLOAD_RANKINGS_PATH = ROOT / "reports/rankings/workload_decision_rankings.csv"
SOURCE_PREDICTIONS_PATH = ROOT / "reports/predictions/production_sources_baseline_predictions.csv"
OPERATIONAL_RANKING_HISTORY_PATH = ROOT / "reports/monitoring/operational_ranking_history.csv"
CHAMPION_PATH = ROOT / "reports/metrics/champion_model_selection.json"
RANKING_METRICS_PATH = ROOT / "reports/metrics/ranking_specific_metrics.json"
CARBON_METRICS_PATH = ROOT / "reports/metrics/carbon_forecast_metrics.json"
RECOMMENDATION_DRIFT_PATH = ROOT / "reports/metrics/future_recommendation_drift_metrics.json"
EMISSION_FACTORS_PATH = ROOT / "config/emission_factors.yaml"
MONITORING_THRESHOLDS_PATH = ROOT / "config/monitoring_thresholds.yaml"

DEFAULT_RECENT_DAYS = 14
SOURCE_TARGETS = ("nuclear", "gas", "coal", "oil", "wind", "solar", "hydro", "bioenergy")


def build_forecast_monitoring_report(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    recent_days: int | None = None,
    thresholds_path: str | Path = MONITORING_THRESHOLDS_PATH,
) -> dict[str, Any]:
    """Build and persist forecast monitoring and retraining trigger report."""
    thresholds = load_monitoring_thresholds(thresholds_path)
    window_days = recent_days if recent_days is not None else int(thresholds["recent_window_days"])
    pipeline_health = load_optional_json(PIPELINE_HEALTH_PATH)
    champion = load_optional_json(CHAMPION_PATH)
    champion_model = champion.get("champion_model")
    features = load_features()
    latest_actual_timestamp = features[TIMESTAMP_COLUMN].max() if not features.empty else None
    cutoff = latest_actual_timestamp - pd.Timedelta(days=window_days) if latest_actual_timestamp is not None else None

    historical = monitor_historical_rankings(champion_model, cutoff)
    operational = monitor_operational_rankings(features, champion_model, cutoff)
    source_drift = monitor_source_prediction_drift(champion_model, cutoff)
    recommendation_drift = monitor_recommendation_drift()
    references = load_reference_metrics(champion_model)
    trigger = evaluate_retraining_trigger(
        pipeline_health=pipeline_health,
        historical=historical,
        operational=operational,
        source_drift=source_drift,
        recommendation_drift=recommendation_drift,
        references=references,
        thresholds=thresholds,
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": trigger["status"],
        "retraining_recommended": trigger["retraining_recommended"],
        "reasons": trigger["reasons"],
        "warnings": trigger["warnings"],
        "recent_window_days": window_days,
        "thresholds": thresholds,
        "champion_model": champion_model,
        "latest_actual_timestamp_utc": latest_actual_timestamp.isoformat()
        if latest_actual_timestamp is not None
        else None,
        "references": references,
        "historical_recent": historical,
        "operational_settled": operational,
        "source_prediction_drift": source_drift,
        "recommendation_drift": recommendation_drift,
        "pipeline_health": summarize_pipeline_health(pipeline_health),
    }
    write_json(output_path, report)
    return report


def load_features() -> pd.DataFrame:
    """Load modeling features with actual observed values."""
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(FEATURES_PATH, parse_dates=[TIMESTAMP_COLUMN])
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    return frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def monitor_historical_rankings(champion_model: str | None, cutoff: pd.Timestamp | None) -> dict[str, Any]:
    """Monitor recent historical validation rows for ranking quality."""
    if not WORKLOAD_RANKINGS_PATH.exists() or not champion_model or cutoff is None:
        return unavailable("historical ranking inputs unavailable")
    rankings = pd.read_csv(WORKLOAD_RANKINGS_PATH, parse_dates=[TIMESTAMP_COLUMN])
    rankings[TIMESTAMP_COLUMN] = pd.to_datetime(rankings[TIMESTAMP_COLUMN], utc=True)
    frame = rankings[
        (rankings["model"] == champion_model)
        & (rankings[TIMESTAMP_COLUMN] >= cutoff)
    ].copy()
    if frame.empty:
        return unavailable("no recent historical ranking rows")
    predicted_best = frame[frame["predicted_decision_rank"] == 1]
    return {
        "available": True,
        "rows": int(len(frame)),
        "decision_groups": int(frame[["window", "decision_group"]].drop_duplicates().shape[0]),
        "top_1_hit_rate": safe_mean(predicted_best["is_actual_best"]),
        "top_5_hit_rate": safe_mean(predicted_best["actual_decision_rank"] <= 5),
        "mean_combined_regret": safe_mean(predicted_best["combined_regret"]),
        "mean_carbon_regret_g_co2e_per_kwh": safe_mean(
            predicted_best["carbon_regret_g_co2e_per_kwh"]
        ),
        "price_direction_accuracy": safe_mean(frame["price_direction_correct"]),
    }


def monitor_operational_rankings(
    features: pd.DataFrame,
    champion_model: str | None,
    cutoff: pd.Timestamp | None,
) -> dict[str, Any]:
    """Compare settled operational forecast history against actual data."""
    if (
        not OPERATIONAL_RANKING_HISTORY_PATH.exists()
        or features.empty
        or not champion_model
        or cutoff is None
    ):
        return unavailable("operational ranking history or actuals unavailable")
    try:
        history = pd.read_csv(OPERATIONAL_RANKING_HISTORY_PATH, parse_dates=[TIMESTAMP_COLUMN])
    except pd.errors.ParserError as error:
        return unavailable(f"operational ranking history malformed: {error}")
    if TIMESTAMP_COLUMN not in history or "model" not in history:
        return unavailable("operational ranking history malformed: missing required columns")
    history[TIMESTAMP_COLUMN] = pd.to_datetime(
        history[TIMESTAMP_COLUMN],
        utc=True,
        errors="coerce",
    )
    history = history.dropna(subset=[TIMESTAMP_COLUMN])
    if history.empty:
        return unavailable("operational ranking history malformed: no valid timestamps")
    history = history[
        (history["model"] == champion_model)
        & (history[TIMESTAMP_COLUMN] >= cutoff)
        & (history[TIMESTAMP_COLUMN] <= features[TIMESTAMP_COLUMN].max())
    ].copy()
    if history.empty:
        return unavailable("no settled operational forecasts overlap actual data yet")

    latest_generation = (
        history.sort_values("forecast_generated_at_utc")
        .drop_duplicates(["forecast_generated_at_utc", "model", "decision_group", TIMESTAMP_COLUMN], keep="last")
        .copy()
    )
    actuals = build_actual_decision_actuals(features)
    settled = latest_generation.merge(actuals, on=TIMESTAMP_COLUMN, how="inner", suffixes=("", "_observed"))
    if settled.empty:
        return unavailable("settled forecasts did not join to actual rows")
    settled = recompute_operational_actual_ranks(settled)
    predicted_best = settled[settled["predicted_decision_rank"] == 1]
    return {
        "available": True,
        "rows": int(len(settled)),
        "settled_forecast_generations": int(settled["forecast_generated_at_utc"].nunique()),
        "decision_groups": int(settled[["forecast_generated_at_utc", "decision_group"]].drop_duplicates().shape[0]),
        "top_1_hit_rate": safe_mean(predicted_best["is_actual_best_observed"]),
        "top_5_hit_rate": safe_mean(predicted_best["actual_decision_rank_observed"] <= 5),
        "carbon_mae_g_co2e_per_kwh": mae(
            predicted_best["predicted_avg_carbon_intensity_g_co2e_per_kwh"]
            - predicted_best["actual_carbon_intensity_g_co2e_per_kwh_observed"]
        ),
        "price_mae_eur_mwh": mae(
            predicted_best["predicted_avg_price_eur_mwh"]
            - predicted_best["actual_price_eur_mwh_observed"]
        ),
        "mean_combined_regret": safe_mean(predicted_best["combined_regret_observed"]),
        "mean_carbon_regret_g_co2e_per_kwh": safe_mean(
            predicted_best["carbon_regret_g_co2e_per_kwh_observed"]
        ),
        "price_direction_accuracy": safe_mean(predicted_best["price_direction_correct_observed"]),
    }


def build_actual_decision_actuals(features: pd.DataFrame) -> pd.DataFrame:
    """Build actual price and carbon intensity for observed timestamps."""
    factors = load_emission_factor_config(EMISSION_FACTORS_PATH)["direct_operational_emissions"]
    output = features[[TIMESTAMP_COLUMN, "price_eur_mwh", "total_production_mwh"]].copy()
    source_columns = [f"{source}_mwh" for source in SOURCE_TARGETS]
    generation = features[source_columns].clip(lower=0)
    emissions = sum(generation[f"{source}_mwh"] * factors[source] for source in SOURCE_TARGETS)
    output["actual_total_emissions_kg_co2e_observed"] = emissions
    output["actual_carbon_intensity_g_co2e_per_kwh_observed"] = (
        emissions / features["total_production_mwh"].replace(0, np.nan)
    )
    output = output.rename(
        columns={
            "price_eur_mwh": "actual_price_eur_mwh_observed",
            "total_production_mwh": "actual_total_generation_mwh_observed",
        }
    )
    previous_day = features[[TIMESTAMP_COLUMN, "price_eur_mwh"]].copy()
    previous_day[TIMESTAMP_COLUMN] = previous_day[TIMESTAMP_COLUMN] + pd.Timedelta(days=1)
    output = output.merge(
        previous_day.rename(columns={"price_eur_mwh": "previous_day_price_eur_mwh_observed"}),
        on=TIMESTAMP_COLUMN,
        how="left",
    )
    return output


def recompute_operational_actual_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute actual ranks/regret for settled operational forecasts."""
    output = frame.copy()
    group_columns = ["forecast_generated_at_utc", "window", "model", "decision_group"]
    output["actual_price_rank_observed"] = rank(output, group_columns, "actual_price_eur_mwh_observed")
    output["actual_carbon_rank_observed"] = rank(
        output,
        group_columns,
        "actual_carbon_intensity_g_co2e_per_kwh_observed",
    )
    output["candidate_count_observed"] = output.groupby(group_columns)[TIMESTAMP_COLUMN].transform("size")
    output["actual_price_rank_pct_observed"] = normalized_rank(
        output["actual_price_rank_observed"],
        output["candidate_count_observed"],
    )
    output["actual_carbon_rank_pct_observed"] = normalized_rank(
        output["actual_carbon_rank_observed"],
        output["candidate_count_observed"],
    )
    output["actual_combined_score_observed"] = (
        0.5 * output["actual_price_rank_pct_observed"]
        + 0.5 * output["actual_carbon_rank_pct_observed"]
    )
    output["actual_decision_rank_observed"] = rank(
        output,
        group_columns,
        "actual_combined_score_observed",
    )
    output["actual_best_score_observed"] = output.groupby(group_columns)[
        "actual_combined_score_observed"
    ].transform("min")
    output["actual_best_carbon_observed"] = output.groupby(group_columns)[
        "actual_carbon_intensity_g_co2e_per_kwh_observed"
    ].transform("min")
    output["combined_regret_observed"] = (
        output["actual_combined_score_observed"] - output["actual_best_score_observed"]
    )
    output["carbon_regret_g_co2e_per_kwh_observed"] = (
        output["actual_carbon_intensity_g_co2e_per_kwh_observed"]
        - output["actual_best_carbon_observed"]
    )
    output["is_actual_best_observed"] = output["actual_decision_rank_observed"] == 1
    predicted_change = output["predicted_avg_price_eur_mwh"] - output[
        "previous_day_price_eur_mwh_observed"
    ]
    actual_change = output["actual_price_eur_mwh_observed"] - output[
        "previous_day_price_eur_mwh_observed"
    ]
    output["price_direction_correct_observed"] = (
        predicted_change.map(direction_label) == actual_change.map(direction_label)
    ).astype(float)
    output.loc[output["previous_day_price_eur_mwh_observed"].isna(), "price_direction_correct_observed"] = np.nan
    return output


def monitor_source_prediction_drift(champion_model: str | None, cutoff: pd.Timestamp | None) -> dict[str, Any]:
    """Monitor recent source-production forecast error drift."""
    if not SOURCE_PREDICTIONS_PATH.exists() or not champion_model or cutoff is None:
        return unavailable("source prediction inputs unavailable")
    predictions = pd.read_csv(SOURCE_PREDICTIONS_PATH, parse_dates=[TIMESTAMP_COLUMN])
    predictions[TIMESTAMP_COLUMN] = pd.to_datetime(predictions[TIMESTAMP_COLUMN], utc=True)
    model_frame = predictions[predictions["model"] == champion_model].copy()
    recent = model_frame[model_frame[TIMESTAMP_COLUMN] >= cutoff].copy()
    if model_frame.empty or recent.empty:
        return unavailable("no recent source prediction rows for champion model")
    return {
        "available": True,
        "rows": int(len(recent)),
        "recent_smape": smape(recent["actual_mwh"], recent["predicted_mwh"]),
        "reference_smape": smape(model_frame["actual_mwh"], model_frame["predicted_mwh"]),
        "by_source": summarize_source_smape(recent),
    }


def monitor_recommendation_drift() -> dict[str, Any]:
    """Load recommendation drift metrics for threshold checks."""
    drift = load_optional_json(RECOMMENDATION_DRIFT_PATH)
    if not drift:
        return unavailable("recommendation drift metrics unavailable")
    recommendations = drift.get("recommendations", {})
    return {
        "available": True,
        "generated_at_utc": drift.get("generated_at_utc"),
        "high_uncertainty_share": recommendations.get("high_uncertainty_share"),
        "average_confidence_score": recommendations.get("average_confidence_score"),
        "recommendation_status_counts": recommendations.get("recommendation_status_counts", {}),
        "rows": recommendations.get("rows", 0),
        "rank_overlap_with_previous": drift.get("rank_overlap_with_previous"),
    }


def summarize_source_smape(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize recent source sMAPE by production source."""
    rows = []
    for source, source_frame in frame.groupby("target", observed=True):
        rows.append(
            {
                "source": str(source),
                "rows": int(len(source_frame)),
                "smape": smape(source_frame["actual_mwh"], source_frame["predicted_mwh"]),
            }
        )
    return sorted(rows, key=lambda row: row["smape"], reverse=True)


def load_reference_metrics(champion_model: str | None) -> dict[str, Any]:
    """Load baseline reference metrics for degradation comparisons."""
    ranking_metrics = load_optional_json(RANKING_METRICS_PATH)
    carbon_metrics = load_optional_json(CARBON_METRICS_PATH)
    ranking = find_model_row(ranking_metrics.get("summary", []), champion_model)
    carbon = find_model_row(
        [
            row
            for row in carbon_metrics.get("summary", [])
            if row.get("methodology") == "direct_operational_emissions"
        ],
        champion_model,
    )
    return {
        "top_5_hit_rate": ranking.get("top_5_recall"),
        "mean_carbon_regret_g_co2e_per_kwh": ranking.get(
            "mean_top_1_carbon_regret_g_co2e_per_kwh"
        ),
        "price_direction_accuracy": ranking.get("price_direction_accuracy"),
        "carbon_intensity_mae_g_co2e_per_kwh": carbon.get(
            "carbon_intensity_mae_g_co2e_per_kwh"
        ),
    }


def evaluate_retraining_trigger(
    pipeline_health: dict[str, Any],
    historical: dict[str, Any],
    operational: dict[str, Any],
    source_drift: dict[str, Any],
    recommendation_drift: dict[str, Any],
    references: dict[str, Any],
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    """Evaluate monitoring rules and return retraining recommendation status."""
    reasons: list[str] = []
    warnings: list[str] = []
    if pipeline_health.get("status") == "fail":
        reasons.append("pipeline_health_failed")

    if operational.get("available"):
        if int(operational.get("rows", 0)) < int(thresholds["min_operational_rows"]):
            warnings.append("operational_settled_rows_below_minimum")
        else:
            compare_greater(
                reasons,
                "operational_carbon_mae_degraded",
                operational.get("carbon_mae_g_co2e_per_kwh"),
                references.get("carbon_intensity_mae_g_co2e_per_kwh"),
                float(thresholds["degradation_ratio"]),
            )
            compare_greater(
                reasons,
                "operational_carbon_regret_degraded",
                operational.get("mean_carbon_regret_g_co2e_per_kwh"),
                references.get("mean_carbon_regret_g_co2e_per_kwh"),
                float(thresholds["degradation_ratio"]),
            )
            compare_drop(
                reasons,
                "operational_top5_hit_rate_dropped",
                operational.get("top_5_hit_rate"),
                references.get("top_5_hit_rate"),
                float(thresholds["top5_drop_threshold"]),
            )
            if not meets_minimum(
                operational.get("price_direction_accuracy"),
                float(thresholds["min_price_direction_accuracy"]),
            ):
                reasons.append("operational_price_direction_accuracy_below_50pct")
    else:
        warnings.append(str(operational.get("reason", "operational monitoring unavailable")))

    if historical.get("available"):
        compare_greater(
            reasons,
            "historical_recent_carbon_regret_degraded",
            historical.get("mean_carbon_regret_g_co2e_per_kwh"),
            references.get("mean_carbon_regret_g_co2e_per_kwh"),
            float(thresholds["degradation_ratio"]),
        )
        compare_drop(
            reasons,
            "historical_recent_top5_hit_rate_dropped",
            historical.get("top_5_hit_rate"),
            references.get("top_5_hit_rate"),
            float(thresholds["top5_drop_threshold"]),
        )

    if source_drift.get("available"):
        compare_greater(
            reasons,
            "source_generation_smape_degraded",
            source_drift.get("recent_smape"),
            source_drift.get("reference_smape"),
            float(thresholds["source_smape_degradation_ratio"]),
        )

    if recommendation_drift.get("available"):
        compare_absolute_greater(
            reasons,
            "recommendation_high_uncertainty_share_high",
            recommendation_drift.get("high_uncertainty_share"),
            float(thresholds["max_high_uncertainty_share"]),
        )
        no_low_risk_share = recommendation_status_share(
            recommendation_drift.get("recommendation_status_counts", {}),
            int(recommendation_drift.get("rows", 0)),
            "no_low_risk_recommendation_available",
        )
        compare_absolute_greater(
            reasons,
            "recommendation_no_low_risk_share_high",
            no_low_risk_share,
            float(thresholds["max_no_low_risk_recommendation_share"]),
        )
        if not meets_minimum(
            recommendation_drift.get("average_confidence_score"),
            float(thresholds["min_average_confidence_score"]),
        ):
            reasons.append("recommendation_average_confidence_low")
        rank_overlap = recommendation_drift.get("rank_overlap_with_previous")
        if rank_overlap is not None and not meets_minimum(
            rank_overlap,
            float(thresholds["min_rank_overlap_with_previous"]),
        ):
            reasons.append("recommendation_rank_overlap_low")
    else:
        warnings.append(str(recommendation_drift.get("reason", "recommendation drift unavailable")))

    return {
        "status": "warn" if reasons or warnings else "pass",
        "retraining_recommended": bool(reasons),
        "reasons": reasons,
        "warnings": warnings,
    }


def compare_greater(
    reasons: list[str],
    reason: str,
    current: float | None,
    reference: float | None,
    ratio: float,
) -> None:
    """Append reason when current is materially greater than reference."""
    if current is None or reference is None or pd.isna(current) or pd.isna(reference):
        return
    if reference <= 0:
        return
    if current > reference * ratio:
        reasons.append(reason)


def compare_drop(
    reasons: list[str],
    reason: str,
    current: float | None,
    reference: float | None,
    drop: float,
) -> None:
    """Append reason when current drops materially below reference."""
    if current is None or reference is None or pd.isna(current) or pd.isna(reference):
        return
    if current < reference - drop:
        reasons.append(reason)


def compare_absolute_greater(
    reasons: list[str],
    reason: str,
    current: float | None,
    threshold: float,
) -> None:
    """Append reason when current exceeds an absolute threshold."""
    if current is None or pd.isna(current):
        return
    if current > threshold:
        reasons.append(reason)


def recommendation_status_share(
    status_counts: dict[str, Any],
    rows: int,
    status: str,
) -> float | None:
    """Return a status share from drift status counts."""
    if rows <= 0:
        return None
    return float(status_counts.get(status, 0)) / rows


def meets_minimum(current: float | None, minimum: float) -> bool:
    """Return whether a metric clears a minimum threshold."""
    if current is None or pd.isna(current):
        return True
    return bool(current >= minimum)


def find_model_row(rows: list[dict[str, Any]], model: str | None) -> dict[str, Any]:
    """Return the first metrics row for a model."""
    for row in rows:
        if row.get("model") == model:
            return row
    return {}


def summarize_pipeline_health(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact pipeline health fields for monitoring."""
    return {
        "status": report.get("status"),
        "critical_issue_count": report.get("critical_issue_count", 0),
        "warning_count": report.get("warning_count", 0),
        "generated_at_utc": report.get("generated_at_utc"),
    }


def unavailable(reason: str) -> dict[str, Any]:
    """Return unavailable monitor section."""
    return {"available": False, "reason": reason}


def rank(frame: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.Series:
    """Rank ascending values within groups."""
    return (
        frame.groupby(group_columns, observed=True)[value_column]
        .rank(method="first", ascending=True)
        .astype(int)
    )


def normalized_rank(rank_values: pd.Series, candidate_count: pd.Series) -> pd.Series:
    """Normalize ranks to [0, 1]."""
    denominator = (candidate_count - 1).replace(0, np.nan)
    return ((rank_values - 1) / denominator).fillna(0.0)


def direction_label(change: float | None) -> str:
    """Convert a price change into a direction label."""
    if pd.isna(change):
        return "unknown"
    if change > 0:
        return "increase"
    if change < 0:
        return "decrease"
    return "flat"


def mae(errors: pd.Series) -> float:
    """Return mean absolute error."""
    value = errors.abs().mean()
    return round(float(value), 4) if not pd.isna(value) else 0.0


def smape(actual: pd.Series, predicted: pd.Series) -> float:
    """Return symmetric mean absolute percentage error."""
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").astype(float)
    denominator = (actual_values.abs() + predicted_values.abs()) / 2
    values = (actual_values - predicted_values).abs() / denominator.replace(0, np.nan)
    value = values.mean()
    return round(float(value), 4) if not pd.isna(value) else 0.0


def safe_mean(values: pd.Series) -> float:
    """Return JSON-safe mean."""
    value = values.astype(float).mean()
    return round(float(value), 4) if not pd.isna(value) else 0.0


def load_optional_json(path: str | Path) -> dict[str, Any]:
    """Load JSON if present."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_monitoring_thresholds(path: str | Path = MONITORING_THRESHOLDS_PATH) -> dict[str, float | int]:
    """Load monitoring thresholds with defaults for missing keys."""
    defaults: dict[str, float | int] = {
        "recent_window_days": DEFAULT_RECENT_DAYS,
        "min_operational_rows": 12,
        "degradation_ratio": 1.25,
        "top5_drop_threshold": 0.15,
        "min_price_direction_accuracy": 0.50,
        "source_smape_degradation_ratio": 1.25,
        "max_high_uncertainty_share": 0.30,
        "max_no_low_risk_recommendation_share": 0.10,
        "min_average_confidence_score": 0.50,
        "min_rank_overlap_with_previous": 0.50,
    }
    threshold_path = Path(path)
    if not threshold_path.exists():
        return defaults
    loaded = yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
    return {**defaults, **loaded}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON output with parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Run forecast monitoring from the command line."""
    parser = argparse.ArgumentParser(description="Monitor forecast quality and drift.")
    parser.add_argument("--recent-days", type=int, default=None)
    parser.add_argument("--thresholds-path", default=str(MONITORING_THRESHOLDS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    report = build_forecast_monitoring_report(
        output_path=args.output_path,
        recent_days=args.recent_days,
        thresholds_path=args.thresholds_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "retraining_recommended": report["retraining_recommended"],
                "output": str(DEFAULT_OUTPUT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
