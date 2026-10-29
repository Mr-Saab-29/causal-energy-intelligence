"""Supabase-backed transformed ingestion for scheduled refreshes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

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
    latest_price_timestamp_utc: str | None
    latest_mix_timestamp_utc: str | None
    latest_feature_timestamp_utc: str | None
    feature_freshness_lag_hours: float | None
    recent_missing_feature_hours: int


MAX_FEATURE_FRESHNESS_HOURS = 48
RECENT_COMPLETENESS_DAYS = 7


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
        refresh_completed=True,
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

    source_timestamps = latest_source_timestamps(engine)
    feature_health = evaluate_feature_freshness(
        features,
        as_of=_freshness_reference(end_date),
    )
    _log(
        "latest timestamps "
        f"price={source_timestamps['latest_price_timestamp_utc']} "
        f"mix={source_timestamps['latest_mix_timestamp_utc']} "
        f"features={feature_health['latest_feature_timestamp_utc']} "
        f"lag_hours={feature_health['feature_freshness_lag_hours']} "
        f"recent_missing_hours={feature_health['recent_missing_feature_hours']}"
    )
    if feature_health["feature_freshness_lag_hours"] is None:
        raise RuntimeError("Modeling feature export contains no valid timestamps.")
    if feature_health["feature_freshness_lag_hours"] > MAX_FEATURE_FRESHNESS_HOURS:
        raise RuntimeError(
            "Modeling features remain stale after ingestion: "
            f"latest_price={source_timestamps['latest_price_timestamp_utc']} "
            f"latest_mix={source_timestamps['latest_mix_timestamp_utc']} "
            f"latest_feature={feature_health['latest_feature_timestamp_utc']} "
            f"lag_hours={feature_health['feature_freshness_lag_hours']}"
        )

    return SupabaseIngestionSummary(
        start_date=target_start_date.isoformat(),
        end_date=target_end_date.isoformat(),
        electricity_price_rows=electricity_price_rows,
        hourly_electricity_mix_rows=hourly_electricity_mix_rows,
        weather_rows=weather_rows,
        feature_rows_written=int(len(features)),
        **source_timestamps,
        **feature_health,
    )


def latest_source_timestamps(engine: Engine) -> dict[str, str | None]:
    """Return the newest timestamps that can participate in the feature join."""
    statement = text(
        """
        select
            (select max(timestamp_utc)
             from electricity_prices
             where region = 'FR' and market = 'day_ahead') as latest_price_timestamp_utc,
            (select max(timestamp_utc)
             from hourly_electricity_mix
             where region = 'FR' and scope = 'national'
               and consumption_mwh is not null
               and total_production_mwh is not null) as latest_mix_timestamp_utc
        """
    )
    with engine.begin() as connection:
        row = connection.execute(statement).mappings().one()
    return {
        key: _isoformat_timestamp(row[key])
        for key in ("latest_price_timestamp_utc", "latest_mix_timestamp_utc")
    }


def evaluate_feature_freshness(
    features: pd.DataFrame,
    *,
    as_of: datetime,
) -> dict[str, str | float | int | None]:
    """Summarize joined-feature freshness and recent internal UTC gaps."""
    if "timestamp_utc" not in features:
        return {
            "latest_feature_timestamp_utc": None,
            "feature_freshness_lag_hours": None,
            "recent_missing_feature_hours": 0,
        }
    timestamps = pd.to_datetime(features["timestamp_utc"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return {
            "latest_feature_timestamp_utc": None,
            "feature_freshness_lag_hours": None,
            "recent_missing_feature_hours": 0,
        }
    timestamps = pd.DatetimeIndex(timestamps.drop_duplicates().sort_values())
    latest = timestamps.max()
    recent_start = max(timestamps.min(), latest - pd.Timedelta(days=RECENT_COMPLETENESS_DAYS))
    expected = pd.date_range(recent_start.ceil("h"), latest.floor("h"), freq="h", tz="UTC")
    missing_hours = int(len(expected.difference(timestamps)))
    lag_hours = max(0.0, (as_of - latest.to_pydatetime()).total_seconds() / 3_600)
    return {
        "latest_feature_timestamp_utc": latest.isoformat(),
        "feature_freshness_lag_hours": round(lag_hours, 2),
        "recent_missing_feature_hours": missing_hours,
    }


def _freshness_reference(explicit_end_date: date | None) -> datetime:
    if explicit_end_date is None:
        return datetime.now(UTC)
    return datetime.combine(explicit_end_date + timedelta(days=1), time.min, tzinfo=UTC)


def _isoformat_timestamp(value: object) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


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
