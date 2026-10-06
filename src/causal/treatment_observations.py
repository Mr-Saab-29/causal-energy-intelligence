"""Readiness checks for observed workload decisions and execution outcomes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from src.data.load import create_database_engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/observed_treatment_readiness.json"
TABLE_NAME = "observed_workload_decisions"
VIEW_NAME = "observed_workload_treatment_analysis"


def load_treatment_observations(engine: Engine) -> pd.DataFrame:
    """Load the decision-level analysis view without direct user identifiers."""
    names = set(inspect(engine).get_table_names(schema="public"))
    views = set(inspect(engine).get_view_names(schema="public"))
    if TABLE_NAME not in names or VIEW_NAME not in views:
        raise RuntimeError(
            "Observed-treatment migration is incomplete. "
            "Apply db/observed_workload_treatment.sql."
        )
    with engine.connect() as connection:
        return pd.read_sql_query(
            text(f"select * from {VIEW_NAME} order by dashboard_accessed_at_utc"),
            connection,
        )


def build_treatment_readiness(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize collection coverage without claiming causal identification."""
    if frame.empty:
        return empty_report()
    data = frame.copy()
    for column in (
        "dashboard_accessed_at_utc",
        "user_selected_start_utc",
        "actual_start_utc",
        "actual_completion_utc",
    ):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    selected = data["user_selected_start_utc"].notna()
    completed = (
        data["actual_start_utc"].notna()
        & data["actual_completion_utc"].notna()
        & pd.to_numeric(data["actual_energy_kwh"], errors="coerce").gt(0)
    )
    completed_rows = data[completed]
    invalid_intervals = int(
        (
            data["actual_completion_utc"].notna()
            & (data["actual_completion_utc"] <= data["actual_start_utc"])
        ).sum()
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "collecting",
        "schema_version": "observed_workload_decision_v1",
        "observed_treatment_available": bool(completed.any()),
        "identified_estimator_ready": False,
        "counts": {
            "decisions": int(len(data)),
            "selected_starts": int(selected.sum()),
            "completed_executions": int(completed.sum()),
            "distinct_decision_days": int(
                data["dashboard_accessed_at_utc"].dt.date.nunique()
            ),
        },
        "coverage": {
            "selection_share": round(float(selected.mean()), 4),
            "completion_share": round(float(completed.mean()), 4),
            "actual_energy_share": round(
                float(data["actual_energy_kwh"].notna().mean()), 4
            ),
        },
        "baseline_source_counts": value_counts(data, "baseline_source"),
        "workload_type_counts": value_counts(data, "workload_type"),
        "selection_source_counts": value_counts(data, "selection_source"),
        "quality": {
            "invalid_execution_intervals": invalid_intervals,
            "completed_energy_kwh": round(
                float(
                    pd.to_numeric(
                        completed_rows["actual_energy_kwh"], errors="coerce"
                    ).sum()
                ),
                3,
            ),
        },
        "remaining_requirements": [
            "time_blocked_overlap_and_positivity_assessment",
            "minimum_sample_size_from_empirical_power_analysis",
            "workload_telemetry_quality_validation",
            "estimator_and_sensitivity_diagnostics",
        ],
    }


def empty_report() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "collecting",
        "schema_version": "observed_workload_decision_v1",
        "observed_treatment_available": False,
        "identified_estimator_ready": False,
        "counts": {
            "decisions": 0,
            "selected_starts": 0,
            "completed_executions": 0,
            "distinct_decision_days": 0,
        },
        "coverage": {
            "selection_share": 0.0,
            "completion_share": 0.0,
            "actual_energy_share": 0.0,
        },
        "baseline_source_counts": {},
        "workload_type_counts": {},
        "selection_source_counts": {},
        "quality": {
            "invalid_execution_intervals": 0,
            "completed_energy_kwh": 0.0,
        },
        "remaining_requirements": [
            "completed_workload_observations",
            "time_blocked_overlap_and_positivity_assessment",
            "minimum_sample_size_from_empirical_power_analysis",
            "workload_telemetry_quality_validation",
            "estimator_and_sensitivity_diagnostics",
        ],
    }


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].dropna().value_counts().to_dict().items()
    }


def write_treatment_readiness(
    engine: Engine,
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    report = build_treatment_readiness(load_treatment_observations(engine))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Audit observed workload treatment collection readiness."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")
    report = write_treatment_readiness(
        create_database_engine(args.database_url), args.output_path
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "observed_treatment_available": report[
                    "observed_treatment_available"
                ],
                "counts": report["counts"],
                "output": args.output_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

