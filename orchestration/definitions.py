"""Dagster assets for the clean-hour dashboard refresh."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from dagster import (
    Definitions,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)

from src.data.pipeline_health import DEFAULT_OUTPUT_PATH, build_pipeline_health
from src.monitoring.forecast_monitor import (
    DEFAULT_OUTPUT_PATH as FORECAST_MONITOR_OUTPUT_PATH,
    build_forecast_monitoring_report,
)

ROOT = Path(__file__).resolve().parents[1]


@asset(group_name="daily_refresh")
def ingest_latest_source_data(context) -> dict[str, Any]:
    """Fetch missing local source data and rebuild modeling features."""
    result = run_command(["make", "ingest-latest"])
    context.add_output_metadata({"stdout_tail": MetadataValue.text(result[-2_000:])})
    return {"command": "make ingest-latest", "stdout_tail": result[-2_000:]}


@asset(group_name="daily_refresh", deps=[ingest_latest_source_data])
def source_data_snapshot(context) -> dict[str, Any]:
    """Validate the source files currently used by the static MVP refresh."""
    report = build_pipeline_health(DEFAULT_OUTPUT_PATH, include_future=False)

    context.add_output_metadata(
        {
            "status": report["status"],
            "critical_issue_count": report["critical_issue_count"],
            "warning_count": report["warning_count"],
            "pipeline_health": MetadataValue.path(str(DEFAULT_OUTPUT_PATH)),
        }
    )
    if report["critical_issue_count"]:
        raise ValueError(f"Pipeline health failed: {report['critical_issue_count']} critical issues")
    return report


@asset(group_name="daily_refresh", deps=[source_data_snapshot])
def clean_hour_forecast_artifacts(context) -> dict[str, Any]:
    """Run the end-to-end clean-hour recommendation pipeline."""
    result = run_command(["make", "train-all-gated"])
    context.add_output_metadata({"stdout_tail": MetadataValue.text(result[-2_000:])})
    return {"command": "make train-all-gated", "stdout_tail": result[-2_000:]}


@asset(group_name="quick_refresh", deps=[source_data_snapshot])
def quick_recommendation_artifacts(context) -> dict[str, Any]:
    """Rebuild recommendations from existing forecast artifacts."""
    result = run_command(["make", "forecast-decision"])
    context.add_output_metadata({"stdout_tail": MetadataValue.text(result[-2_000:])})
    return {"command": "make forecast-decision", "stdout_tail": result[-2_000:]}


@asset(group_name="daily_refresh", deps=[clean_hour_forecast_artifacts])
def future_exogenous_data(context) -> dict[str, Any]:
    """Fetch future exogenous inputs for operational recommendations."""
    result = run_command(["make", "ingest-future"])
    output_path = ROOT / "data/processed/future_weather_forecast.csv"
    context.add_output_metadata(
        {
            "future_weather": MetadataValue.path(str(output_path)),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "stdout_tail": MetadataValue.text(result[-2_000:]),
        }
    )
    return {"command": "make ingest-future", "stdout_tail": result[-2_000:]}


@asset(group_name="quick_refresh", deps=[source_data_snapshot])
def quick_future_exogenous_data(context) -> dict[str, Any]:
    """Fetch future exogenous inputs without retraining historical artifacts."""
    result = run_command(["make", "ingest-future"])
    output_path = ROOT / "data/processed/future_weather_forecast.csv"
    context.add_output_metadata(
        {
            "future_weather": MetadataValue.path(str(output_path)),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "stdout_tail": MetadataValue.text(result[-2_000:]),
        }
    )
    return {"command": "make ingest-future", "stdout_tail": result[-2_000:]}


@asset(group_name="monitoring", deps=[quick_future_exogenous_data])
def forecast_monitoring_report(context) -> dict[str, Any]:
    """Compare settled forecasts with actuals and flag retraining needs."""
    build_pipeline_health(DEFAULT_OUTPUT_PATH, include_future=True)
    report = build_forecast_monitoring_report(FORECAST_MONITOR_OUTPUT_PATH)
    context.add_output_metadata(
        {
            "status": report["status"],
            "retraining_recommended": report["retraining_recommended"],
            "reason_count": len(report["reasons"]),
            "forecast_monitoring": MetadataValue.path(str(FORECAST_MONITOR_OUTPUT_PATH)),
        }
    )
    return report


@asset(group_name="daily_refresh", deps=[future_exogenous_data])
def future_recommendation_artifacts(context) -> dict[str, Any]:
    """Build operational next-24-hour recommendations from saved artifacts."""
    result = run_command(["make", "future-recommendations"])
    output_path = ROOT / "reports/recommendations/future_champion_workload_recommendations.csv"
    context.add_output_metadata(
        {
            "future_recommendations": MetadataValue.path(str(output_path)),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "stdout_tail": MetadataValue.text(result[-2_000:]),
        }
    )
    return {"command": "make future-recommendations", "stdout_tail": result[-2_000:]}


@asset(group_name="quick_refresh", deps=[quick_future_exogenous_data])
def quick_future_recommendation_artifacts(context) -> dict[str, Any]:
    """Build operational recommendations without retraining historical artifacts."""
    result = run_command(["make", "future-recommendations"])
    output_path = ROOT / "reports/recommendations/future_champion_workload_recommendations.csv"
    context.add_output_metadata(
        {
            "future_recommendations": MetadataValue.path(str(output_path)),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "stdout_tail": MetadataValue.text(result[-2_000:]),
        }
    )
    return {"command": "make future-recommendations", "stdout_tail": result[-2_000:]}


@asset(group_name="daily_refresh", deps=[future_recommendation_artifacts])
def dashboard_data_contract(context) -> dict[str, Any]:
    """Build the static JSON data contract consumed by the dashboard."""
    result = run_command(["make", "dashboard-data"])
    output_path = ROOT / "frontend/public/data/dashboard.json"
    context.add_output_metadata(
        {
            "dashboard_json": MetadataValue.path(str(output_path)),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        }
    )
    return {"command": "make dashboard-data", "stdout_tail": result[-2_000:]}


@asset(group_name="quick_refresh", deps=[quick_future_recommendation_artifacts])
def quick_dashboard_data_contract(context) -> dict[str, Any]:
    """Build the dashboard JSON after a quick recommendation refresh."""
    result = run_command(["make", "dashboard-data"])
    output_path = ROOT / "frontend/public/data/dashboard.json"
    context.add_output_metadata(
        {
            "dashboard_json": MetadataValue.path(str(output_path)),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        }
    )
    return {"command": "make dashboard-data", "stdout_tail": result[-2_000:]}


@asset(group_name="daily_refresh", deps=[dashboard_data_contract])
def frontend_static_build(context) -> dict[str, Any]:
    """Build the Vercel-ready frontend bundle from the latest dashboard data."""
    result = run_command(["make", "frontend-build"])
    output_path = ROOT / "frontend/dist/index.html"
    context.add_output_metadata(
        {
            "index_html": MetadataValue.path(str(output_path)),
            "exists": output_path.exists(),
        }
    )
    return {"command": "make frontend-build", "stdout_tail": result[-2_000:]}


@asset(group_name="quick_refresh", deps=[quick_dashboard_data_contract])
def quick_frontend_static_build(context) -> dict[str, Any]:
    """Build the frontend bundle after a quick recommendation refresh."""
    result = run_command(["make", "frontend-build"])
    output_path = ROOT / "frontend/dist/index.html"
    context.add_output_metadata(
        {
            "index_html": MetadataValue.path(str(output_path)),
            "exists": output_path.exists(),
        }
    )
    return {"command": "make frontend-build", "stdout_tail": result[-2_000:]}


daily_refresh_job = define_asset_job(
    name="daily_clean_hour_refresh",
    selection=[
        "ingest_latest_source_data",
        "source_data_snapshot",
        "clean_hour_forecast_artifacts",
        "future_exogenous_data",
        "future_recommendation_artifacts",
        "dashboard_data_contract",
        "frontend_static_build",
    ],
)

quick_recommendation_refresh = define_asset_job(
    name="quick_recommendation_refresh",
    selection=[
        "ingest_latest_source_data",
        "source_data_snapshot",
        "quick_future_exogenous_data",
        "quick_future_recommendation_artifacts",
        "quick_dashboard_data_contract",
        "quick_frontend_static_build",
    ],
)

ingestion_monitor_refresh = define_asset_job(
    name="ingestion_monitor_refresh",
    selection=[
        "ingest_latest_source_data",
        "source_data_snapshot",
        "quick_future_exogenous_data",
        "forecast_monitoring_report",
    ],
)

daily_refresh_schedule = ScheduleDefinition(
    job=ingestion_monitor_refresh,
    cron_schedule="0 2 * * *",
    execution_timezone="Europe/Paris",
)

defs = Definitions(
    assets=[
        ingest_latest_source_data,
        source_data_snapshot,
        forecast_monitoring_report,
        clean_hour_forecast_artifacts,
        quick_recommendation_artifacts,
        future_exogenous_data,
        quick_future_exogenous_data,
        future_recommendation_artifacts,
        quick_future_recommendation_artifacts,
        dashboard_data_contract,
        quick_dashboard_data_contract,
        frontend_static_build,
        quick_frontend_static_build,
    ],
    jobs=[daily_refresh_job, quick_recommendation_refresh, ingestion_monitor_refresh],
    schedules=[daily_refresh_schedule],
)


def run_command(command: list[str]) -> str:
    """Run a command from the repository root and return stdout/stderr."""
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return "\n".join(part for part in [completed.stdout, completed.stderr] if part)
