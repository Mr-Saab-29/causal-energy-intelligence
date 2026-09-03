"""Build static dashboard data from clean-hour recommendation artifacts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pipeline_health import DEFAULT_OUTPUT_PATH, build_pipeline_health
from src.causal.recommendations import build_causal_adjusted_recommendations
from src.optimization.workload_shift import (
    CONFIDENCE_CALIBRATION_PATH,
    SCENARIO_CONFIDENCE_CALIBRATION_PATH,
    add_recommendation_confidence,
    apply_confidence_calibration,
    build_scenario_rerankings,
    build_top_workload_recommendations,
    load_confidence_calibration,
)

OUTPUT_PATH = ROOT / "frontend/public/data/dashboard.json"
PRODUCTION_MODEL_LABEL = "Production Model V1"


def main() -> None:
    """Write the dashboard JSON payload used by the frontend."""
    champion = read_json(ROOT / "reports/metrics/champion_model_selection.json")
    decision_metrics = read_json(ROOT / "reports/metrics/workload_decision_metrics.json")
    ranking_metrics = read_json(ROOT / "reports/metrics/ranking_specific_metrics.json")
    scenario_metrics = read_json(ROOT / "reports/metrics/scenario_reranking_metrics.json")
    policy_backtest = read_json(ROOT / "reports/metrics/recommendation_policy_backtest.json")
    scenario_champions = read_json(ROOT / "reports/metrics/scenario_champion_selection.json")
    marginal_shift_metrics = read_json(ROOT / "reports/metrics/marginal_ranking_shift_metrics.json")
    forecast_monitoring_path = ROOT / "reports/metrics/forecast_monitoring.json"
    forecast_monitoring = read_json(forecast_monitoring_path)
    recommendation_drift = read_json(ROOT / "reports/metrics/future_recommendation_drift_metrics.json")
    forecast_monitoring_stale = is_forecast_monitoring_stale(forecast_monitoring_path)
    pipeline_health = build_pipeline_health(DEFAULT_OUTPUT_PATH)
    recommendations = read_csv(
        ROOT / "reports/recommendations/champion_workload_recommendations.csv"
    )
    future_recommendations = read_csv(
        ROOT / "reports/recommendations/future_champion_workload_recommendations.csv"
    )
    future_rankings = read_csv(
        ROOT / "reports/rankings/future_workload_decision_rankings.csv"
    )
    future_causal_recommendations = read_csv(
        ROOT / "reports/recommendations/future_causal_adjusted_workload_recommendations.csv"
    )
    future_marginal_rankings = read_csv(
        ROOT / "reports/rankings/future_marginal_workload_decision_rankings.csv"
    )
    active_future_recommendations = build_active_future_recommendations(
        future_recommendations,
        future_rankings,
    )
    active_recommendations = (
        active_future_recommendations
        if not active_future_recommendations.empty
        else pd.DataFrame()
    )
    scenario_recommendations = read_csv(
        ROOT / "reports/scenarios/workload_scenario_recommendations.csv"
    )
    future_scenario_recommendations = read_csv(
        ROOT / "reports/scenarios/future_workload_scenario_recommendations.csv"
    )
    active_future_scenario_recommendations = build_active_future_scenario_recommendations(
        future_scenario_recommendations,
        future_rankings,
    )
    active_scenario_recommendations = (
        active_future_scenario_recommendations
        if not active_future_scenario_recommendations.empty
        else pd.DataFrame()
    )

    active_scenario_recommendations = enrich_scenario_recommendations(
        active_scenario_recommendations,
        active_recommendations,
    )
    active_recommendations = normalize_recommendation_fields(active_recommendations)
    active_scenario_recommendations = normalize_recommendation_fields(
        active_scenario_recommendations
    )
    active_future_causal_recommendations = build_active_future_causal_recommendations(
        future_causal_recommendations,
        future_marginal_rankings,
    )
    active_causal_recommendations = normalize_recommendation_fields(
        active_future_causal_recommendations
    )
    recommendation_rows = prepare_records(active_recommendations)
    scenario_rows = prepare_records(active_scenario_recommendations)
    causal_rows = prepare_records(active_causal_recommendations)
    filter_dates = sorted(
        set(safe_unique(active_recommendations, "decision_group"))
        | set(safe_unique(active_scenario_recommendations, "decision_group"))
        | set(safe_unique(active_causal_recommendations, "decision_group"))
    )
    payload = {
        "generated_from": {
            "champion_model_selection": "reports/metrics/champion_model_selection.json",
            "recommendations": (
                "reports/recommendations/future_champion_workload_recommendations.csv"
                if not active_future_recommendations.empty
                else "reports/recommendations/champion_workload_recommendations.csv"
            ),
            "scenario_recommendations": "reports/scenarios/workload_scenario_recommendations.csv",
            "future_scenario_recommendations": (
                "reports/scenarios/future_workload_scenario_recommendations.csv"
                if not active_future_scenario_recommendations.empty
                else None
            ),
            "recommendation_drift": "reports/metrics/future_recommendation_drift_metrics.json",
            "policy_backtest": "reports/metrics/recommendation_policy_backtest.json",
            "scenario_champion_selection": "reports/metrics/scenario_champion_selection.json",
            "marginal_ranking_shift_metrics": (
                "reports/metrics/marginal_ranking_shift_metrics.json"
                if marginal_shift_metrics
                else None
            ),
            "causal_adjusted_recommendations": (
                "reports/recommendations/future_causal_adjusted_workload_recommendations.csv"
                if not future_causal_recommendations.empty
                else None
            ),
        },
        "champion": {
            "model": champion.get("champion_model"),
            "display_model_name": PRODUCTION_MODEL_LABEL,
            "weights": champion.get("weights", {}),
            "selection_rule": champion.get("selection_rule"),
            "models": champion.get("models", []),
        },
        "summary": {
            "pipeline_health": summarize_pipeline_health(pipeline_health),
            "forecast_monitoring": summarize_forecast_monitoring(
                forecast_monitoring,
                stale=forecast_monitoring_stale,
            ),
            "decision_metrics": decision_metrics.get("summary", []),
            "ranking_metrics": ranking_metrics.get("summary", []),
            "scenario_metrics": scenario_metrics.get("summary", []),
            "policy_backtest": policy_backtest,
            "scenario_champions": scenario_champions.get("champions", []),
            "marginal_ranking_shift": summarize_marginal_shift_metrics(
                marginal_shift_metrics
            ),
            "date_count": int(active_recommendations["decision_group"].nunique())
            if not active_recommendations.empty
            else 0,
            "recommendation_count": int(len(active_recommendations)),
            "future_recommendation_file_rows": int(len(future_recommendations)),
            "active_future_recommendation_count": int(len(active_future_recommendations)),
            "future_scenario_file_rows": int(len(future_scenario_recommendations)),
            "active_future_scenario_count": int(len(active_future_scenario_recommendations)),
            "active_future_causal_count": int(len(active_causal_recommendations)),
            "stale_future_recommendations": bool(
                not future_recommendations.empty and active_future_recommendations.empty
            ),
            "stale_future_scenarios": bool(
                not future_scenario_recommendations.empty
                and active_future_scenario_recommendations.empty
            ),
            "average_confidence_score": safe_float(
                active_recommendations["confidence_score"].mean()
            )
            if "confidence_score" in active_recommendations
            else None,
            "high_confidence_share": safe_float(
                (active_recommendations["confidence_level"] == "high").mean()
            )
            if "confidence_level" in active_recommendations
            else None,
            "high_uncertainty_share": safe_float(
                active_recommendations["decision_uncertainty_score"].gt(0.85).mean()
            )
            if "decision_uncertainty_score" in active_recommendations
            else None,
        },
        "pipeline_health": pipeline_health,
        "forecast_monitoring": {
            **forecast_monitoring,
            "stale": forecast_monitoring_stale,
        },
        "recommendation_drift": recommendation_drift,
        "marginal_ranking_shift": marginal_shift_metrics,
        "filters": {
            "dates": filter_dates,
            "scenarios": safe_unique(active_scenario_recommendations, "scenario"),
        },
        "recommendations": recommendation_rows,
        "scenario_recommendations": scenario_rows,
        "causal_recommendations": causal_rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def filter_future_recommendations(
    frame: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only operational recommendation rows at or after the current UTC hour."""
    if frame.empty or "timestamp_utc" not in frame:
        return frame.iloc[0:0].copy()
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    current_hour = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    return frame.loc[timestamps >= current_hour].copy()


def build_active_future_recommendations(
    recommendations: pd.DataFrame,
    rankings: pd.DataFrame,
    now: pd.Timestamp | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return active top-N recommendations, refilling from future rankings when needed."""
    active = filter_future_recommendations(recommendations, now=now)
    if has_top_n_per_group(active, ["window", "model", "decision_group"], top_n):
        return active
    active_rankings = filter_future_recommendations(rankings, now=now)
    if active_rankings.empty:
        return active
    rebuilt = build_top_workload_recommendations(active_rankings, top_n=top_n)
    rebuilt = add_recommendation_confidence(rebuilt, active_rankings, top_n=top_n)
    return apply_confidence_calibration(
        rebuilt,
        load_confidence_calibration(CONFIDENCE_CALIBRATION_PATH),
    )


def build_active_future_scenario_recommendations(
    scenario_recommendations: pd.DataFrame,
    rankings: pd.DataFrame,
    now: pd.Timestamp | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return active scenario top-N rows, refilling from future rankings when needed."""
    active = filter_future_recommendations(scenario_recommendations, now=now)
    if has_top_n_per_group(active, ["scenario", "window", "model", "decision_group"], top_n):
        return active
    active_rankings = filter_future_recommendations(rankings, now=now)
    if active_rankings.empty:
        return active
    rebuilt, _ = build_scenario_rerankings(active_rankings, top_n=top_n)
    return apply_confidence_calibration(
        rebuilt,
        load_confidence_calibration(SCENARIO_CONFIDENCE_CALIBRATION_PATH),
    )


def build_active_future_causal_recommendations(
    recommendations: pd.DataFrame,
    marginal_rankings: pd.DataFrame,
    now: pd.Timestamp | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return active causal-adjusted top-N rows, preserving marginal proxy context."""
    active = filter_future_recommendations(recommendations, now=now)
    if has_top_n_per_group(active, ["window", "model", "decision_group"], top_n):
        return active
    active_rankings = filter_future_recommendations(marginal_rankings, now=now)
    if active_rankings.empty:
        return active
    return build_causal_adjusted_recommendations(active_rankings, top_n=top_n)


def has_top_n_per_group(frame: pd.DataFrame, group_columns: list[str], top_n: int) -> bool:
    """Return whether every visible group has at least top-N rows."""
    if frame.empty or not all(column in frame for column in group_columns):
        return False
    return bool((frame.groupby(group_columns, dropna=False).size() >= top_n).all())


def safe_unique(frame: pd.DataFrame, column: str) -> list[Any]:
    """Return sorted unique non-null values when a column exists."""
    if frame.empty or column not in frame:
        return []
    return sorted(frame[column].dropna().unique().tolist())


def prepare_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON records with stable null handling."""
    if frame.empty:
        return []
    cleaned = frame.replace({pd.NA: None})
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return sanitize_json_value(cleaned.to_dict(orient="records"))


def sanitize_json_value(value: Any) -> Any:
    """Return a value that can be emitted as strict JSON."""
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return sanitize_json_value(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def enrich_scenario_recommendations(
    scenario_frame: pd.DataFrame,
    recommendation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add shared recommendation context to scenario reranking rows."""
    if scenario_frame.empty or recommendation_frame.empty:
        return scenario_frame
    join_columns = ["decision_group", "timestamp_utc"]
    if not all(column in scenario_frame for column in join_columns):
        return scenario_frame
    if not all(column in recommendation_frame for column in join_columns):
        return scenario_frame

    context_columns = [
        "candidate_count",
        "confidence_score",
        "confidence_level",
        "heuristic_confidence_score",
        "heuristic_confidence_level",
        "empirical_top_n_hit_rate",
        "expected_carbon_regret_g_co2e_per_kwh",
        "expected_cost_regret_eur_mwh",
    ]
    available_context_columns = [
        column
        for column in context_columns
        if column in recommendation_frame and column not in scenario_frame
    ]
    if not available_context_columns:
        return scenario_frame

    context = recommendation_frame[join_columns + available_context_columns].drop_duplicates(
        subset=join_columns
    )
    return scenario_frame.merge(context, on=join_columns, how="left")


def normalize_recommendation_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill optional recommendation fields expected by the dashboard."""
    if frame.empty:
        return frame
    output = frame.copy()
    if "recommendation_status" not in output:
        output["recommendation_status"] = "recommended"
    output["recommendation_status"] = output["recommendation_status"].fillna("recommended")
    if "suppressed_by_uncertainty_guard" not in output:
        output["suppressed_by_uncertainty_guard"] = False
    output["suppressed_by_uncertainty_guard"] = output[
        "suppressed_by_uncertainty_guard"
    ].fillna(False)
    if "decision_uncertainty_score" not in output:
        output["decision_uncertainty_score"] = None
    rank_group_columns = [
        column
        for column in ["scenario", "window", "model", "decision_group"]
        if column in output
    ]
    if "recommendation_rank" in output and rank_group_columns:
        output = output.sort_values(
            rank_group_columns + ["recommendation_rank", "timestamp_utc"],
            na_position="last",
        ).copy()
        output["recommendation_rank"] = (
            output.groupby(rank_group_columns, dropna=False).cumcount() + 1
        )
    return output


def safe_float(value: float) -> float | None:
    """Return a rounded JSON-safe float."""
    if pd.isna(value):
        return None
    return round(float(value), 4)


def summarize_pipeline_health(report: dict[str, Any]) -> dict[str, Any]:
    """Return the compact health summary shown in the dashboard."""
    latest_timestamps = [
        source.get("max_timestamp_utc")
        for source in report.get("sources", {}).values()
        if source.get("max_timestamp_utc")
    ]
    return {
        "status": report.get("status"),
        "generated_at_utc": report.get("generated_at_utc"),
        "critical_issue_count": report.get("critical_issue_count", 0),
        "warning_count": report.get("warning_count", 0),
        "latest_data_timestamp_utc": max(latest_timestamps) if latest_timestamps else None,
    }


def summarize_forecast_monitoring(report: dict[str, Any], stale: bool = False) -> dict[str, Any]:
    """Return compact monitoring fields for dashboard status."""
    return {
        "status": "stale" if stale else report.get("status", "unknown"),
        "generated_at_utc": report.get("generated_at_utc"),
        "retraining_recommended": bool(report.get("retraining_recommended", False)) and not stale,
        "stale": stale,
        "reason_count": len(report.get("reasons", [])),
        "warning_count": len(report.get("warnings", [])),
        "latest_actual_timestamp_utc": report.get("latest_actual_timestamp_utc"),
        "champion_model": report.get("champion_model"),
    }


def summarize_marginal_shift_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact causal-adjusted MVP metrics for the dashboard."""
    future = report.get("future", report) if isinstance(report, dict) else {}
    aggregate = future.get("aggregate", {}) if isinstance(future, dict) else {}
    quality_guard = future.get("quality_guard", {}) if isinstance(future, dict) else {}
    return {
        "method": future.get("method"),
        "quality_status": quality_guard.get("status", "unknown"),
        "warnings": quality_guard.get("warnings", []),
        "top_1_change_share": safe_float(aggregate.get("top_1_change_share")),
        "mean_top_5_overlap_share": safe_float(aggregate.get("mean_top_5_overlap_share")),
        "mean_absolute_rank_shift": safe_float(aggregate.get("mean_absolute_rank_shift")),
        "mean_causal_adjustment_coverage": safe_float(
            aggregate.get("mean_causal_adjustment_coverage")
        ),
        "mean_top_1_regret_delta": safe_float(aggregate.get("mean_top_1_regret_delta")),
    }


def is_forecast_monitoring_stale(path: Path) -> bool:
    """Return whether monitoring output is older than model-quality inputs."""
    if not path.exists():
        return False
    watched_paths = [
        ROOT / "reports/metrics/champion_model_selection.json",
        ROOT / "reports/metrics/ranking_specific_metrics.json",
        ROOT / "reports/metrics/carbon_forecast_metrics.json",
        ROOT / "reports/rankings/workload_decision_rankings.csv",
    ]
    monitor_mtime = path.stat().st_mtime
    return any(
        watched_path.exists() and watched_path.stat().st_mtime > monitor_mtime
        for watched_path in watched_paths
    )


if __name__ == "__main__":
    main()
