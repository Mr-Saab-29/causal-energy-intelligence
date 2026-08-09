"""Local pipeline health checks for clean-hour scheduling artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "reports/metrics/pipeline_health.json"
TIMESTAMP_COLUMN = "timestamp_utc"
DEFAULT_MAX_FRESHNESS_DAYS = 2


@dataclass(frozen=True)
class SourceHealthConfig:
    """Configuration for one local CSV health check."""

    name: str
    path: str
    timestamp_column: str = TIMESTAMP_COLUMN
    required_columns: tuple[str, ...] = ()
    non_negative_columns: tuple[str, ...] = ()
    group_columns: tuple[str, ...] = ()
    nullable_columns: tuple[str, ...] = ()
    max_freshness_days: int = DEFAULT_MAX_FRESHNESS_DAYS


SOURCE_CONFIGS = [
    SourceHealthConfig(
        name="electricity_prices",
        path="data/processed/electricity_prices.csv",
        required_columns=("timestamp_utc", "price_eur_mwh"),
        group_columns=("region", "market"),
    ),
    SourceHealthConfig(
        name="hourly_electricity_mix",
        path="data/processed/hourly_electricity_mix.csv",
        required_columns=(
            "timestamp_utc",
            "consumption_mwh",
            "total_production_mwh",
            "nuclear_mwh",
            "gas_mwh",
            "wind_mwh",
            "solar_mwh",
            "hydro_mwh",
            "bioenergy_mwh",
        ),
        non_negative_columns=(
            "consumption_mwh",
            "total_production_mwh",
            "nuclear_mwh",
            "gas_mwh",
            "coal_mwh",
            "oil_mwh",
            "wind_mwh",
            "solar_mwh",
            "hydro_mwh",
            "bioenergy_mwh",
        ),
        group_columns=("region", "scope"),
        nullable_columns=("gas_mwh", "coal_mwh", "oil_mwh"),
    ),
    SourceHealthConfig(
        name="weather_observations",
        path="data/processed/weather_observations.csv",
        required_columns=(
            "timestamp_utc",
            "temperature_c",
            "wind_speed_mps",
            "shortwave_radiation_wm2",
        ),
        non_negative_columns=(
            "wind_speed_mps",
            "shortwave_radiation_wm2",
            "precipitation_mm",
        ),
        group_columns=("region",),
    ),
    SourceHealthConfig(
        name="modeling_price_features",
        path="data/processed/modeling_price_features.csv",
        required_columns=(
            "timestamp_utc",
            "price_eur_mwh",
            "consumption_mwh",
            "total_production_mwh",
            "nuclear_mwh",
            "gas_mwh",
            "wind_mwh",
            "solar_mwh",
            "hydro_mwh",
            "bioenergy_mwh",
        ),
        non_negative_columns=(
            "consumption_mwh",
            "total_production_mwh",
            "nuclear_mwh",
            "gas_mwh",
            "coal_mwh",
            "oil_mwh",
            "wind_mwh",
            "solar_mwh",
            "hydro_mwh",
            "bioenergy_mwh",
        ),
    ),
]


def build_pipeline_health(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    source_configs: list[SourceHealthConfig] | None = None,
    strict_freshness: bool = True,
) -> dict[str, Any]:
    """Build and persist a JSON health report for local source and dashboard artifacts."""
    configs = source_configs or SOURCE_CONFIGS
    sources = {
        config.name: check_source(config, strict_freshness=strict_freshness)
        for config in configs
    }
    dashboard = check_dashboard_artifacts()
    critical_issue_count = sum(len(source["critical_issues"]) for source in sources.values())
    warning_count = sum(len(source["warnings"]) for source in sources.values()) + len(
        dashboard["warnings"]
    )
    if dashboard["critical_issues"]:
        critical_issue_count += len(dashboard["critical_issues"])

    status = "fail" if critical_issue_count else "warn" if warning_count else "pass"
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "critical_issue_count": critical_issue_count,
        "warning_count": warning_count,
        "sources": sources,
        "dashboard": dashboard,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def check_source(config: SourceHealthConfig, strict_freshness: bool = True) -> dict[str, Any]:
    """Check one local CSV source file."""
    path = ROOT / config.path
    result: dict[str, Any] = {
        "path": config.path,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "row_count": 0,
        "min_timestamp_utc": None,
        "max_timestamp_utc": None,
        "freshness_lag_days": None,
        "missing_hour_count": None,
        "duplicate_timestamp_count": None,
        "null_counts": {},
        "negative_value_counts": {},
        "critical_issues": [],
        "warnings": [],
    }
    if not path.exists():
        result["critical_issues"].append("file_missing")
        return result
    if result["size_bytes"] == 0:
        result["critical_issues"].append("file_empty")
        return result

    frame = pd.read_csv(path)
    result["row_count"] = int(len(frame))
    if frame.empty:
        result["critical_issues"].append("no_rows")
        return result

    missing_required = [column for column in config.required_columns if column not in frame.columns]
    if missing_required:
        result["critical_issues"].append(f"missing_required_columns:{','.join(missing_required)}")
    if config.timestamp_column not in frame.columns:
        result["critical_issues"].append(f"missing_timestamp_column:{config.timestamp_column}")
        return result

    timestamps = pd.to_datetime(frame[config.timestamp_column], utc=True, errors="coerce")
    invalid_timestamp_count = int(timestamps.isna().sum())
    if invalid_timestamp_count:
        result["critical_issues"].append(f"invalid_timestamp_count:{invalid_timestamp_count}")
    valid_frame = frame.loc[timestamps.notna()].copy()
    valid_frame[config.timestamp_column] = timestamps[timestamps.notna()]
    if valid_frame.empty:
        result["critical_issues"].append("no_valid_timestamps")
        return result

    min_timestamp = valid_frame[config.timestamp_column].min()
    max_timestamp = valid_frame[config.timestamp_column].max()
    result["min_timestamp_utc"] = min_timestamp.isoformat()
    result["max_timestamp_utc"] = max_timestamp.isoformat()
    result["freshness_lag_days"] = round(
        (datetime.now(UTC) - max_timestamp.to_pydatetime()).total_seconds() / 86_400,
        2,
    )
    if result["freshness_lag_days"] > config.max_freshness_days:
        issue = f"stale_latest_timestamp:{result['freshness_lag_days']}_days"
        if strict_freshness:
            result["critical_issues"].append(issue)
        else:
            result["warnings"].append(issue)

    duplicate_subset = [
        column for column in (*config.group_columns, config.timestamp_column) if column in valid_frame
    ]
    result["duplicate_timestamp_count"] = int(valid_frame.duplicated(duplicate_subset).sum())
    if result["duplicate_timestamp_count"]:
        result["critical_issues"].append(
            f"duplicate_timestamp_count:{result['duplicate_timestamp_count']}"
        )

    result["missing_hour_count"] = count_missing_hours(
        valid_frame,
        config.timestamp_column,
        [column for column in config.group_columns if column in valid_frame],
    )
    if result["missing_hour_count"]:
        result["warnings"].append(f"missing_hour_count:{result['missing_hour_count']}")

    checked_null_columns = [column for column in config.required_columns if column in valid_frame]
    null_counts: dict[str, dict[str, float | int]] = {}
    for column in checked_null_columns:
        null_count = int(valid_frame[column].isna().sum())
        null_counts[column] = {
            "count": null_count,
            "rate": round(null_count / result["row_count"], 4),
        }
        if null_count:
            if column not in config.nullable_columns:
                result["warnings"].append(f"null_count:{column}:{null_count}")
    result["null_counts"] = null_counts

    checked_negative_columns = [
        column for column in config.non_negative_columns if column in valid_frame
    ]
    result["negative_value_counts"] = {
        column: int((pd.to_numeric(valid_frame[column], errors="coerce") < 0).sum())
        for column in checked_negative_columns
    }
    for column, negative_count in result["negative_value_counts"].items():
        if negative_count:
            result["critical_issues"].append(f"negative_value_count:{column}:{negative_count}")

    return result


def count_missing_hours(frame: pd.DataFrame, timestamp_column: str, group_columns: list[str]) -> int:
    """Count missing hourly timestamps across the full observed range."""
    if group_columns:
        return int(
            sum(
                count_missing_hours(group, timestamp_column, [])
                for _, group in frame.groupby(group_columns, observed=True)
            )
        )

    timestamps = frame[timestamp_column].dropna().sort_values().drop_duplicates()
    if len(timestamps) < 2:
        return 0
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h", tz="UTC")
    return int(len(expected.difference(pd.DatetimeIndex(timestamps))))


def check_dashboard_artifacts() -> dict[str, Any]:
    """Check generated dashboard and recommendation artifacts."""
    dashboard_json_path = ROOT / "frontend/public/data/dashboard.json"
    recommendations_path = ROOT / "reports/recommendations/champion_workload_recommendations.csv"
    result: dict[str, Any] = {
        "dashboard_json": file_summary(dashboard_json_path),
        "recommendations": file_summary(recommendations_path),
        "recommendation_count": 0,
        "latest_recommendation_date": None,
        "critical_issues": [],
        "warnings": [],
    }

    if not dashboard_json_path.exists():
        result["warnings"].append("dashboard_json_missing")
    if not recommendations_path.exists():
        result["warnings"].append("recommendations_missing")
        return result

    recommendations = pd.read_csv(recommendations_path)
    result["recommendation_count"] = int(len(recommendations))
    if recommendations.empty:
        result["warnings"].append("recommendations_empty")
        return result
    if "decision_group" in recommendations:
        result["latest_recommendation_date"] = str(
            recommendations["decision_group"].dropna().max()
        )
    return result


def file_summary(path: Path) -> dict[str, Any]:
    """Return a compact file existence/size summary."""
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> None:
    """Build the pipeline health report from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Build local pipeline health report.")
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()
    report = build_pipeline_health(strict_freshness=not args.allow_stale)
    print(json.dumps({"status": report["status"], "output": str(DEFAULT_OUTPUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
