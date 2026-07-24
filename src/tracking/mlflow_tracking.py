"""MLflow tracking helpers for forecast and recommendation runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

TRACKING_ARTIFACTS = [
    "reports/metrics/price_baseline_metrics.json",
    "reports/metrics/supply_demand_baseline_metrics.json",
    "reports/metrics/production_sources_baseline_metrics.json",
    "reports/metrics/carbon_forecast_metrics.json",
    "reports/metrics/workload_decision_metrics.json",
    "reports/metrics/ranking_specific_metrics.json",
    "reports/metrics/champion_model_selection.json",
    "reports/metrics/scenario_reranking_metrics.json",
    "reports/recommendations/champion_workload_recommendations.csv",
    "reports/recommendations/top5_workload_recommendations.csv",
    "reports/scenarios/workload_scenario_recommendations.csv",
    "frontend/public/data/dashboard.json",
]


def track_forecast_run(
    target: str,
    result: dict[str, Any],
    params: dict[str, Any],
    artifact_paths: list[str] | None = None,
) -> None:
    """Log forecast/recommendation run metadata and artifacts to MLflow if enabled."""
    if os.getenv("MLFLOW_TRACKING_ENABLED", "true").lower() in {"0", "false", "no"}:
        return

    try:
        import mlflow
    except ImportError:
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "clean-hour-scheduling")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"forecast-{target}"):
        mlflow.log_param("target", target)
        for key, value in params.items():
            if value is not None:
                mlflow.log_param(key, value)
        if result.get("champion_model"):
            mlflow.log_param("champion_model_internal", result["champion_model"])

        log_summary_metrics(mlflow, result.get("summary", []))
        for path in artifact_paths or TRACKING_ARTIFACTS:
            artifact = Path(path)
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path=artifact.parent.as_posix())


def log_summary_metrics(mlflow: Any, summary: Any) -> None:
    """Log numeric summary metrics with model-safe names."""
    if not isinstance(summary, list):
        return
    for row in summary:
        if not isinstance(row, dict):
            continue
        prefix_parts = [str(row[key]) for key in ("target", "model") if row.get(key)]
        prefix = ".".join(prefix_parts) if prefix_parts else "summary"
        prefix = normalize_metric_name(prefix)
        for key, value in row.items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"{prefix}.{normalize_metric_name(key)}", float(value))


def normalize_metric_name(value: str) -> str:
    """Normalize strings for MLflow metric names."""
    return (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_")
        .lower()
    )
