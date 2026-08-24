"""Supabase-backed transformed ingestion for scheduled refreshes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

from src.data.load import create_database_engine
from src.data.pipelines.supabase_load import (
    load_france_electricity_mix_realtime,
    load_france_price_history,
    load_france_weather_history,
)
from src.features.price_features import export_price_modeling_features


@dataclass(frozen=True)
class SupabaseIngestionSummary:
    """Summary of one transformed Supabase ingestion refresh."""

    start_date: str
    end_date: str
    electricity_price_rows: int
    hourly_electricity_mix_rows: int
    weather_rows: int
    feature_rows_written: int


def refresh_supabase_model_data(
    database_url: str,
    start_date: date | None = None,
    end_date: date | None = None,
    lookback_days: int = 45,
    include_regional_mix: bool = False,
) -> SupabaseIngestionSummary:
    """Fetch recent source data, load transformed rows, and export model features."""
    target_end_date = end_date or datetime.now(UTC).date()
    target_start_date = start_date or (
        target_end_date - timedelta(days=max(lookback_days, 0))
    )
    _log(f"refresh window start={target_start_date} end={target_end_date}")
    engine = create_database_engine(database_url)

    _log("loading energy-charts prices")
    electricity_price_rows = load_france_price_history(
        engine=engine,
        start_date=target_start_date,
        end_date=target_end_date,
    )
    _log(f"prices complete rows={electricity_price_rows}")

    _log("loading ODRE production/consumption mix")
    hourly_electricity_mix_rows = load_france_electricity_mix_realtime(
        engine=engine,
        start_date=target_start_date,
        end_date=target_end_date,
        include_regional=include_regional_mix,
    )
    _log(f"production/consumption mix complete rows={hourly_electricity_mix_rows}")

    _log("loading Open-Meteo historical weather")
    weather_rows = load_france_weather_history(
        engine=engine,
        start_date=target_start_date,
        end_date=target_end_date,
    )
    _log(f"weather complete rows={weather_rows}")

    _log("building local modeling feature cache")
    features = export_price_modeling_features(engine)
    _log(f"modeling features complete rows={len(features)}")

    return SupabaseIngestionSummary(
        start_date=target_start_date.isoformat(),
        end_date=target_end_date.isoformat(),
        electricity_price_rows=electricity_price_rows,
        hourly_electricity_mix_rows=hourly_electricity_mix_rows,
        weather_rows=weather_rows,
        feature_rows_written=int(len(features)),
    )


def parse_date(value: str | None) -> date | None:
    """Parse YYYY-MM-DD CLI values."""
    return date.fromisoformat(value) if value else None


def _log(message: str) -> None:
    print(f"[supabase-ingest] {message}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run transformed-only Supabase ingestion from the command line."""
    parser = argparse.ArgumentParser(description="Refresh transformed model data in Supabase.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--include-regional-mix", action="store_true")
    args = parser.parse_args(argv)

    if not args.database_url:
        raise SystemExit("DATABASE_URL is required for Supabase ingestion.")

    summary = refresh_supabase_model_data(
        database_url=args.database_url,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        lookback_days=args.lookback_days,
        include_regional_mix=args.include_regional_mix,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
