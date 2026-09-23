"""Quality and temporal-integrity checks for ENTSO-E causal-grid data."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.entsoe_causal_ingest import (
    REQUIRED_CAUSAL_GRID_TABLES,
    validate_causal_grid_schema,
)
from src.data.load import create_database_engine
from src.data.source_config import FRANCE_DIRECT_ENTSOE_NEIGHBORS, FRANCE_START_DATE

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "reports/metrics/causal_data_quality.json"
REQUIRED_FORECAST_TYPES = ("load", "wind_onshore", "wind_offshore", "solar")
REQUIRED_CROSS_BORDER_METRICS = ("physical_flow", "scheduled_exchange")
REQUIRED_BALANCING_METRICS = (
    "imbalance_price",
    "imbalance_volume",
    "activated_energy_price",
)
GRANULARITY_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440}
DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_MAX_PROJECTED_STORAGE_BYTES = 250 * 1024**2


def build_causal_data_quality_report(
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    vintage_quality: str = "historical_final",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_projected_storage_bytes: int = DEFAULT_MAX_PROJECTED_STORAGE_BYTES,
    output_path: str | Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """Collect, evaluate, and persist causal-grid quality metrics."""
    start = normalize_boundary(start)
    end = normalize_boundary(end)
    if end <= start:
        raise ValueError("end must be after start")
    if vintage_quality not in {"historical_final", "operational_snapshot"}:
        raise ValueError("unsupported vintage_quality")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if max_projected_storage_bytes <= 0:
        raise ValueError("max_projected_storage_bytes must be positive")

    validate_causal_grid_schema(engine)
    metrics = collect_causal_data_metrics(engine, start, end, vintage_quality)
    metrics["storage_projection"] = estimate_storage_projection(metrics, start, end)
    evaluation = evaluate_causal_data_metrics(
        metrics,
        start,
        end,
        vintage_quality=vintage_quality,
        min_coverage=min_coverage,
        max_projected_storage_bytes=max_projected_storage_bytes,
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": evaluation["status"],
        "backfill_ready": evaluation["backfill_ready"],
        "window": {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "vintage_quality": vintage_quality,
            "min_coverage": min_coverage,
            "max_projected_storage_bytes": max_projected_storage_bytes,
        },
        "critical_issues": evaluation["critical_issues"],
        "warnings": evaluation["warnings"],
        "known_limitations": evaluation["known_limitations"],
        "coverage": evaluation["coverage"],
        "metrics": metrics,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def collect_causal_data_metrics(
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    vintage_quality: str,
) -> dict[str, Any]:
    """Collect bounded aggregate metrics without loading raw rows into memory."""
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": vintage_quality,
    }
    with engine.connect() as connection:
        cross_border = _mapping_rows(
            connection,
            """
            select metric, from_bidding_zone, to_bidding_zone, granularity,
                   count(*) as row_count,
                   count(distinct timestamp_utc) as timestamp_count
            from cross_border_observations
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = :vintage
            group by metric, from_bidding_zone, to_bidding_zone, granularity
            order by metric, from_bidding_zone, to_bidding_zone, granularity
            """,
            params,
        )
        forecasts = _mapping_rows(
            connection,
            """
            select forecast_type, granularity,
                   count(*) as row_count,
                   count(distinct timestamp_utc) as timestamp_count,
                   count(*) filter (where forecast_mw < 0) as negative_value_count
            from grid_forecasts
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = :vintage
            group by forecast_type, granularity
            order by forecast_type, granularity
            """,
            params,
        )
        outages = _mapping_rows(
            connection,
            """
            select outage_type, count(*) as row_count,
                   count(distinct outage_mrid) as outage_count,
                   count(*) filter (where unavailable_capacity_mw is null)
                       as missing_unavailable_capacity_count
            from generation_outages
            where timestamp_utc < :end and end_utc >= :start
              and vintage_quality = :vintage
            group by outage_type
            order by outage_type
            """,
            params,
        )
        balancing = _mapping_rows(
            connection,
            """
            select metric, granularity,
                   count(*) as row_count,
                   count(distinct timestamp_utc) as timestamp_count
            from balancing_observations
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = :vintage
            group by metric, granularity
            order by metric, granularity
            """,
            params,
        )
        duplicate_source_keys = {
            table: int(
                connection.execute(
                    text(
                        f"""
                        select count(*) from (
                            select source, source_record_id
                            from {table}
                            group by source, source_record_id
                            having count(*) > 1
                        ) duplicates
                        """
                    )
                ).scalar_one()
            )
            for table in REQUIRED_CAUSAL_GRID_TABLES
        }
        temporal_violations = _temporal_violations(connection, params)
        storage_bytes = _table_storage_bytes(connection)
        table_row_counts = {
            table: int(
                connection.execute(text(f"select count(*) from {table}")).scalar_one()
            )
            for table in REQUIRED_CAUSAL_GRID_TABLES
        }

    return {
        "cross_border_series": cross_border,
        "forecast_series": forecasts,
        "outage_summary": outages,
        "balancing_series": balancing,
        "duplicate_source_key_groups": duplicate_source_keys,
        "temporal_violations": temporal_violations,
        "storage_bytes": storage_bytes,
        "table_row_counts": table_row_counts,
        "total_storage_bytes": sum(storage_bytes.values()),
    }


def evaluate_causal_data_metrics(
    metrics: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    vintage_quality: str,
    min_coverage: float,
    max_projected_storage_bytes: int | None = None,
) -> dict[str, Any]:
    """Apply blocking and advisory policy to collected aggregate metrics."""
    critical: list[str] = []
    warnings: list[str] = []
    known_limitations = [
        "activated_energy_unavailable_from_legacy_a83_endpoint",
    ]
    if vintage_quality == "historical_final":
        known_limitations.append(
            "historical_final_rows_are_not_point_in_time_operational_vintages"
        )

    forecast_coverage = _series_coverage(
        metrics.get("forecast_series", []),
        start,
        end,
        ("forecast_type",),
    )
    for forecast_type in REQUIRED_FORECAST_TYPES:
        ratio = forecast_coverage.get(forecast_type)
        if ratio is None:
            critical.append(f"missing_forecast_type:{forecast_type}")
        elif ratio < min_coverage:
            critical.append(f"low_forecast_coverage:{forecast_type}:{ratio}")

    cross_coverage = _series_coverage(
        metrics.get("cross_border_series", []),
        start,
        end,
        ("metric", "from_bidding_zone", "to_bidding_zone"),
    )
    for metric in REQUIRED_CROSS_BORDER_METRICS:
        for neighbor in FRANCE_DIRECT_ENTSOE_NEIGHBORS:
            for source, target in (("FR", neighbor), (neighbor, "FR")):
                key = f"{metric}|{source}|{target}"
                ratio = cross_coverage.get(key)
                if ratio is None:
                    critical.append(f"missing_cross_border_series:{key}")
                elif ratio < min_coverage:
                    critical.append(f"low_cross_border_coverage:{key}:{ratio}")

    capacity_pairs = {
        "|".join((row["from_bidding_zone"], row["to_bidding_zone"]))
        for row in metrics.get("cross_border_series", [])
        if row["metric"] == "day_ahead_capacity"
    }
    missing_capacity_pairs = []
    for neighbor in FRANCE_DIRECT_ENTSOE_NEIGHBORS:
        for source, target in (("FR", neighbor), (neighbor, "FR")):
            key = f"{source}|{target}"
            if key not in capacity_pairs:
                missing_capacity_pairs.append(key)
    if missing_capacity_pairs:
        known_limitations.append(
            "day_ahead_capacity_not_published_for_pairs:"
            + ",".join(missing_capacity_pairs)
        )

    balancing_coverage = _series_coverage(
        metrics.get("balancing_series", []),
        start,
        end,
        ("metric",),
    )
    for metric in REQUIRED_BALANCING_METRICS:
        ratio = balancing_coverage.get(metric)
        if ratio is None:
            critical.append(f"missing_balancing_metric:{metric}")
        elif ratio < min_coverage:
            critical.append(f"low_balancing_coverage:{metric}:{ratio}")

    outage_rows = sum(
        int(row["row_count"]) for row in metrics.get("outage_summary", [])
    )
    if outage_rows == 0:
        warnings.append("no_outages_overlapping_window")
    other_outages = sum(
        int(row["row_count"])
        for row in metrics.get("outage_summary", [])
        if row["outage_type"] == "other"
    )
    if other_outages:
        critical.append(f"unclassified_outage_rows:{other_outages}")

    for table, count in metrics.get("duplicate_source_key_groups", {}).items():
        if count:
            critical.append(f"duplicate_source_keys:{table}:{count}")
    for check, count in metrics.get("temporal_violations", {}).items():
        if count:
            critical.append(f"temporal_violation:{check}:{count}")

    projected_storage = int(
        metrics.get("storage_projection", {}).get("total_projected_storage_bytes", 0)
    )
    if (
        max_projected_storage_bytes is not None
        and projected_storage > max_projected_storage_bytes
    ):
        critical.append(
            "projected_storage_exceeds_budget:"
            f"{projected_storage}>{max_projected_storage_bytes}"
        )

    status = "fail" if critical else "warn" if warnings else "pass"
    return {
        "status": status,
        "backfill_ready": not critical,
        "critical_issues": critical,
        "warnings": warnings,
        "known_limitations": known_limitations,
        "coverage": {
            "forecasts": forecast_coverage,
            "cross_border": cross_coverage,
            "balancing": balancing_coverage,
        },
    }


def estimate_storage_projection(
    metrics: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Project storage for the configured historical horizon from the pilot window."""
    window_days = max(1.0, (end - start).total_seconds() / 86_400)
    target_start = pd.Timestamp(FRANCE_START_DATE, tz="UTC")
    target_end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    target_days = max(1.0, (target_end - target_start).total_seconds() / 86_400)
    window_rows = {
        "cross_border_observations": sum(
            int(row["row_count"])
            for row in metrics.get("cross_border_series", [])
        ),
        "grid_forecasts": sum(
            int(row["row_count"]) for row in metrics.get("forecast_series", [])
        ),
        "generation_outages": sum(
            int(row["row_count"]) for row in metrics.get("outage_summary", [])
        ),
        "balancing_observations": sum(
            int(row["row_count"]) for row in metrics.get("balancing_series", [])
        ),
    }
    tables: dict[str, dict[str, float | int]] = {}
    for table in REQUIRED_CAUSAL_GRID_TABLES:
        current_rows = int(metrics.get("table_row_counts", {}).get(table, 0))
        current_bytes = int(metrics.get("storage_bytes", {}).get(table, 0))
        bytes_per_row = current_bytes / current_rows if current_rows else 0.0
        rows_per_day = window_rows[table] / window_days
        projected_rows = int(round(rows_per_day * target_days))
        projected_bytes = int(round(projected_rows * bytes_per_row))
        tables[table] = {
            "pilot_window_rows": window_rows[table],
            "estimated_rows_per_day": round(rows_per_day, 2),
            "observed_bytes_per_row": round(bytes_per_row, 2),
            "projected_rows": projected_rows,
            "projected_storage_bytes": projected_bytes,
        }
    return {
        "history_start_date": target_start.date().isoformat(),
        "projection_end_date": target_end.date().isoformat(),
        "projected_days": int(target_days),
        "tables": tables,
        "total_projected_storage_bytes": sum(
            int(table["projected_storage_bytes"]) for table in tables.values()
        ),
    }


def default_quality_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the previous complete UTC calendar month."""
    current_month = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    return current_month - pd.offsets.MonthBegin(1), current_month


def normalize_boundary(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _series_coverage(
    rows: list[dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    key_columns: tuple[str, ...],
) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for row in rows:
        minutes = GRANULARITY_MINUTES.get(str(row["granularity"]))
        if minutes is None:
            continue
        expected = int((end - start).total_seconds() // (minutes * 60))
        ratio = min(1.0, int(row["timestamp_count"]) / expected) if expected else 0.0
        key = "|".join(str(row[column]) for column in key_columns)
        coverage[key] = max(coverage.get(key, 0.0), round(ratio, 4))
    return coverage


def _temporal_violations(connection: Any, params: dict[str, Any]) -> dict[str, int]:
    queries = {
        "operational_forecast_after_target": """
            select count(*) from grid_forecasts
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = 'operational_snapshot'
              and (forecast_generated_at_utc is null
                   or forecast_generated_at_utc > timestamp_utc)
        """,
        "operational_schedule_after_target": """
            select count(*) from cross_border_observations
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = 'operational_snapshot'
              and metric in ('scheduled_exchange', 'day_ahead_capacity')
              and snapshot_at_utc > timestamp_utc
        """,
        "operational_outage_published_after_snapshot": """
            select count(*) from generation_outages
            where timestamp_utc < :end and end_utc >= :start
              and vintage_quality = 'operational_snapshot'
              and publication_timestamp_utc > snapshot_at_utc
        """,
        "historical_forecast_claims_generation_time": """
            select count(*) from grid_forecasts
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = 'historical_final'
              and forecast_generated_at_utc is not null
        """,
        "historical_cross_border_has_snapshot_suffix": """
            select count(*) from cross_border_observations
            where timestamp_utc >= :start and timestamp_utc < :end
              and vintage_quality = 'historical_final'
              and source_record_id like '%:historical_final:%'
        """,
    }
    return {
        name: int(connection.execute(text(sql), params).scalar_one())
        for name, sql in queries.items()
    }


def _table_storage_bytes(connection: Any) -> dict[str, int]:
    return {
        table: int(
            connection.execute(
                text("select pg_total_relation_size(to_regclass(:table_name))"),
                {"table_name": f"public.{table}"},
            ).scalar_one()
            or 0
        )
        for table in REQUIRED_CAUSAL_GRID_TABLES
    }


def _mapping_rows(
    connection: Any,
    sql: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(text(sql), params).mappings().all()]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    default_start, default_end = default_quality_window()
    parser = argparse.ArgumentParser(description="Audit ENTSO-E causal-grid data quality.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--start-date", default=default_start.date().isoformat())
    parser.add_argument("--end-date", default=default_end.date().isoformat())
    parser.add_argument(
        "--vintage-quality",
        choices=("historical_final", "operational_snapshot"),
        default="historical_final",
    )
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument(
        "--max-projected-storage-mib",
        type=float,
        default=DEFAULT_MAX_PROJECTED_STORAGE_BYTES / 1024**2,
    )
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")

    report = build_causal_data_quality_report(
        create_database_engine(args.database_url),
        normalize_boundary(args.start_date),
        normalize_boundary(args.end_date),
        vintage_quality=args.vintage_quality,
        min_coverage=args.min_coverage,
        max_projected_storage_bytes=int(args.max_projected_storage_mib * 1024**2),
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "backfill_ready": report["backfill_ready"],
                "critical_issues": report["critical_issues"],
                "warnings": report["warnings"],
                "known_limitations": report["known_limitations"],
                "storage_projection": report["metrics"]["storage_projection"],
                "output": str(Path(args.output_path)),
            },
            indent=2,
        )
    )
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
