"""Ingest ENTSO-E grid-state inputs used by the causal estimator."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from src.data.load import (
    create_database_engine,
    upsert_balancing_observations,
    upsert_cross_border_observations,
    upsert_generation_outages,
    upsert_grid_forecasts,
)
from src.data.source_config import (
    FRANCE_DIRECT_ENTSOE_NEIGHBORS,
    FRANCE_ENTSOE_AREA,
)
from src.data.sources.entsoe_causal import fetch_entsoe_causal_data

REQUIRED_CAUSAL_GRID_TABLES = (
    "cross_border_observations",
    "grid_forecasts",
    "generation_outages",
    "balancing_observations",
)


@dataclass(frozen=True)
class EntsoeCausalIngestionSummary:
    """Machine-readable result for one bounded ingestion run."""

    status: str
    start_utc: str
    end_utc: str
    historical_backfill: bool
    france_area: str
    neighboring_areas: tuple[str, ...]
    cross_border_rows: int = 0
    forecast_rows: int = 0
    outage_rows: int = 0
    balancing_rows: int = 0
    warnings: tuple[str, ...] = ()


def ingest_entsoe_causal_data(
    api_token: str,
    database_url: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    historical_backfill: bool = False,
) -> EntsoeCausalIngestionSummary:
    """Fetch and upsert a bounded ENTSO-E window."""
    engine = create_database_engine(database_url)
    validate_causal_grid_schema(engine)
    data = fetch_entsoe_causal_data(
        api_token,
        start,
        end,
        historical_backfill=historical_backfill,
    )
    return EntsoeCausalIngestionSummary(
        status="ok",
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        historical_backfill=historical_backfill,
        france_area=FRANCE_ENTSOE_AREA,
        neighboring_areas=FRANCE_DIRECT_ENTSOE_NEIGHBORS,
        cross_border_rows=upsert_cross_border_observations(engine, data.cross_border),
        forecast_rows=upsert_grid_forecasts(engine, data.forecasts),
        outage_rows=upsert_generation_outages(engine, data.outages),
        balancing_rows=upsert_balancing_observations(engine, data.balancing),
        warnings=data.warnings,
    )


def validate_causal_grid_schema(engine: Engine) -> None:
    """Fail before API extraction when the causal-grid migration is incomplete."""
    existing = set(inspect(engine).get_table_names(schema="public"))
    missing = [table for table in REQUIRED_CAUSAL_GRID_TABLES if table not in existing]
    if missing:
        raise RuntimeError(
            "Causal-grid database migration is incomplete. Missing tables: "
            f"{', '.join(missing)}. Apply the complete db/causal_grid_data.sql file."
        )


def build_window(
    start_date: str | None,
    end_date: str | None,
    lookback_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Build a half-open UTC extraction window."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC")
    start = _date_boundary(start_date) if start_date else today - timedelta(days=lookback_days)
    end = _date_boundary(end_date) + timedelta(days=1) if end_date else today + timedelta(days=2)
    if end <= start:
        raise ValueError("end date must be on or after start date")
    return start, end


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ingest cross-border, outage, balancing, and grid forecast data."
    )
    parser.add_argument("--api-token", default=os.environ.get("ENTSOE_API_TOKEN"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--historical-backfill", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    start, end = build_window(args.start_date, args.end_date, args.lookback_days)
    if args.plan_only:
        summary = EntsoeCausalIngestionSummary(
            status="planned",
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
            historical_backfill=args.historical_backfill,
            france_area=FRANCE_ENTSOE_AREA,
            neighboring_areas=FRANCE_DIRECT_ENTSOE_NEIGHBORS,
        )
    else:
        if not args.api_token:
            parser.error("ENTSOE_API_TOKEN or --api-token is required")
        if not args.database_url:
            parser.error("DATABASE_URL or --database-url is required")
        summary = ingest_entsoe_causal_data(
            args.api_token,
            args.database_url,
            start,
            end,
            historical_backfill=args.historical_backfill,
        )
    print(json.dumps(asdict(summary), indent=2))


def _date_boundary(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").normalize()


if __name__ == "__main__":
    main()
