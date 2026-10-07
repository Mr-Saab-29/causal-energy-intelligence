"""Compact recent operational ENTSO-E vintages for point-in-time causal adjustment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from src.data.causal_archive_compact import (
    CROSS_BORDER_COMPACTION_SQL,
    FRANCE_COMPACTION_SQL,
    apply_compact_schema,
    validate_compact_schema,
)
from src.data.entsoe_causal_ingest import (
    REQUIRED_CAUSAL_GRID_TABLES,
    validate_causal_grid_schema,
)
from src.data.load import create_database_engine
from src.data.source_config import FRANCE_DIRECT_ENTSOE_NEIGHBORS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_operational_snapshot.json"


def first_snapshot_sql(sql: str, conflict_columns: str, *, france: bool = False) -> str:
    """Keep the first usable operational vintage instead of overwriting it later."""
    statement = sql.split("\non conflict", maxsplit=1)[0]
    if france:
        join = "left join balancing_hourly as balancing using (timestamp_utc)"
        statement = statement.replace(
            join,
            join
            + """
where coalesce(forecast.forecast_interval_count, 0) > 0
   or coalesce(outage.planned_outage_count, 0) > 0
   or coalesce(outage.unplanned_outage_count, 0) > 0
   or coalesce(balancing.balancing_interval_count, 0) > 0""",
        )
    return f"{statement}\non conflict ({conflict_columns}) do nothing"


OPERATIONAL_CROSS_BORDER_SQL = first_snapshot_sql(
    CROSS_BORDER_COMPACTION_SQL,
    "timestamp_utc, neighbor_bidding_zone, vintage_quality",
)
OPERATIONAL_FRANCE_SQL = first_snapshot_sql(
    FRANCE_COMPACTION_SQL,
    "timestamp_utc, vintage_quality",
    france=True,
)


def compact_operational_snapshot(
    database_url: str,
    *,
    lookback_days: int = 2,
    future_days: int = 2,
    purge_raw: bool = True,
) -> dict[str, object]:
    """Compact a bounded operational window and optionally clear staging rows."""
    if lookback_days < 1 or future_days < 1:
        raise ValueError("lookback_days and future_days must be positive")
    today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC")
    start = today - timedelta(days=lookback_days)
    end = today + timedelta(days=future_days)
    engine = create_database_engine(database_url)
    validate_causal_grid_schema(engine)
    apply_compact_schema(engine)
    validate_compact_schema(engine)
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": "operational_snapshot",
    }
    with engine.begin() as connection:
        connection.execute(text(OPERATIONAL_CROSS_BORDER_SQL), params)
        connection.execute(text(OPERATIONAL_FRANCE_SQL), params)
        cross_count = int(
            connection.execute(
                text(
                    """select count(*) from causal_cross_border_hourly_features
                    where timestamp_utc >= :start and timestamp_utc < :end
                      and vintage_quality = :vintage"""
                ),
                params,
            ).scalar_one()
        )
        france_count = int(
            connection.execute(
                text(
                    """select count(*) from causal_france_hourly_features
                    where timestamp_utc >= :start and timestamp_utc < :end
                      and vintage_quality = :vintage"""
                ),
                params,
            ).scalar_one()
        )
    expected_hours = int((end - start).total_seconds() // 3600)
    expected_cross = expected_hours * len(FRANCE_DIRECT_ENTSOE_NEIGHBORS)
    cross_coverage = cross_count / expected_cross if expected_cross else 0.0
    warnings = []
    if france_count < expected_hours:
        warnings.append(f"partial_france_hour_coverage:{france_count / expected_hours:.4f}")
    if cross_count < expected_cross:
        warnings.append(f"partial_cross_border_coverage:{cross_coverage:.4f}")
    purged: dict[str, int] = {}
    if purge_raw:
        with engine.begin() as connection:
            for table in REQUIRED_CAUSAL_GRID_TABLES:
                result = connection.execute(
                    text(
                        f"""delete from {table}
                        where timestamp_utc >= :start and timestamp_utc < :end
                          and vintage_quality = :vintage"""
                    ),
                    params,
                )
                purged[table] = int(result.rowcount or 0)
    return {
        "status": "warn" if warnings else "ok",
        "vintage_quality": "operational_snapshot",
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "compact_rows": {
            "causal_cross_border_hourly_features": cross_count,
            "causal_france_hourly_features": france_count,
        },
        "coverage": {
            "france_hour_share": round(france_count / expected_hours, 4),
            "cross_border_row_share": round(cross_coverage, 4),
        },
        "warnings": warnings,
        "raw_rows_purged": purged,
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compact recent point-in-time causal-grid snapshots."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--lookback-days", type=int, default=2)
    parser.add_argument("--future-days", type=int, default=2)
    parser.add_argument("--output-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")
    report = compact_operational_snapshot(
        args.database_url,
        lookback_days=args.lookback_days,
        future_days=args.future_days,
        purge_raw=not args.keep_raw,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
