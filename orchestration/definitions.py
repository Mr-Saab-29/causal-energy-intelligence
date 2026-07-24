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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_FILES = [
    "data/processed/electricity_prices.csv",
    "data/processed/hourly_electricity_mix.csv",
    "data/processed/weather_observations.csv",
    "data/processed/modeling_price_features.csv",
]


@asset(group_name="daily_refresh")
def source_data_snapshot(context) -> dict[str, Any]:
    """Validate the source files currently used by the static MVP refresh."""
    files = []
    for relative_path in SOURCE_DATA_FILES:
        path = ROOT / relative_path
        files.append(
            {
                "path": relative_path,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    missing = [row["path"] for row in files if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing source data files: {missing}")

    context.add_output_metadata(
        {
            "file_count": len(files),
            "total_size_bytes": sum(row["size_bytes"] for row in files),
        }
    )
    return {"files": files}


@asset(group_name="daily_refresh", deps=[source_data_snapshot])
def clean_hour_forecast_artifacts(context) -> dict[str, Any]:
    """Run the end-to-end clean-hour recommendation pipeline."""
    result = run_command(["make", "forecast-all"])
    context.add_output_metadata({"stdout_tail": MetadataValue.text(result[-2_000:])})
    return {"command": "make forecast-all", "stdout_tail": result[-2_000:]}


@asset(group_name="daily_refresh", deps=[clean_hour_forecast_artifacts])
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


daily_refresh_job = define_asset_job(
    name="daily_clean_hour_refresh",
    selection=[
        "source_data_snapshot",
        "clean_hour_forecast_artifacts",
        "dashboard_data_contract",
        "frontend_static_build",
    ],
)

daily_refresh_schedule = ScheduleDefinition(
    job=daily_refresh_job,
    cron_schedule="0 5 * * *",
    execution_timezone="Europe/Paris",
)

defs = Definitions(
    assets=[
        source_data_snapshot,
        clean_hour_forecast_artifacts,
        dashboard_data_contract,
        frontend_static_build,
    ],
    jobs=[daily_refresh_job],
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
