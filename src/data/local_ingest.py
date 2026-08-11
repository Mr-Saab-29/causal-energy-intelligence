"""Daily local CSV ingestion for the clean-hour scheduling MVP."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.france_regions import get_regional_weather_locations
from src.data.load import HOURLY_ELECTRICITY_MIX_COLUMNS, PRICE_COLUMNS, WEATHER_COLUMNS
from src.data.sources.energy_charts import fetch_energy_charts_day_ahead_prices
from src.data.sources.odre import (
    fetch_france_national_realtime_hourly_mix,
    fetch_france_regional_realtime_hourly_mix,
)
from src.data.sources.open_meteo import fetch_open_meteo_weather_windowed
from src.features.price_features import build_price_modeling_features

ROOT = Path(__file__).resolve().parents[2]
PRICE_PATH = ROOT / "data/processed/electricity_prices.csv"
MIX_PATH = ROOT / "data/processed/hourly_electricity_mix.csv"
WEATHER_PATH = ROOT / "data/processed/weather_observations.csv"
FEATURES_PATH = ROOT / "data/processed/modeling_price_features.csv"
SOURCE_RECORD_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\+00:00|Z)?)")


@dataclass(frozen=True)
class LocalIngestionSummary:
    """Summary of one local ingestion refresh."""

    start_date: str
    end_date: str
    dry_run: bool
    price_rows_fetched: int = 0
    mix_rows_fetched: int = 0
    weather_rows_fetched: int = 0
    feature_rows_written: int = 0


def refresh_local_data(
    end_date: date | None = None,
    start_date: date | None = None,
    lookback_days: int | None = None,
    dry_run: bool = False,
    include_prices: bool = True,
    include_mix: bool = True,
    include_weather: bool = True,
    rebuild_features: bool = True,
    plan_only: bool = False,
) -> LocalIngestionSummary:
    """Fetch missing local source data and rebuild modeling features."""
    target_end_date = end_date or datetime.now(UTC).date()
    inferred_start_date = resolve_refresh_start_date(
        target_end_date=target_end_date,
        explicit_start_date=start_date,
        lookback_days=lookback_days,
        source_paths=[PRICE_PATH, MIX_PATH, WEATHER_PATH],
    )
    if inferred_start_date > target_end_date:
        inferred_start_date = target_end_date
    if plan_only:
        return LocalIngestionSummary(
            start_date=inferred_start_date.isoformat(),
            end_date=target_end_date.isoformat(),
            dry_run=True,
        )

    price_rows = 0
    mix_rows = 0
    weather_rows = 0
    if include_prices:
        price_observations = fetch_energy_charts_day_ahead_prices(
            inferred_start_date,
            target_end_date,
        )
        price_rows = len(price_observations)
        if not dry_run:
            upsert_observations_csv(
                PRICE_PATH,
                price_observations,
                PRICE_COLUMNS,
                dedupe_columns=["source_record_id"],
            )

    if include_mix:
        mix_observations = [
            *fetch_france_national_realtime_hourly_mix(inferred_start_date, target_end_date),
            *fetch_france_regional_realtime_hourly_mix(inferred_start_date, target_end_date),
        ]
        mix_rows = len(mix_observations)
        if not dry_run:
            upsert_observations_csv(
                MIX_PATH,
                mix_observations,
                HOURLY_ELECTRICITY_MIX_COLUMNS,
                dedupe_columns=["source_record_id"],
            )

    if include_weather:
        weather_observations = []
        for location in get_regional_weather_locations():
            weather_observations.extend(
                fetch_open_meteo_weather_windowed(
                    region=location.region,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    start_date=inferred_start_date,
                    end_date=target_end_date,
                )
            )
        weather_rows = len(weather_observations)
        if not dry_run:
            upsert_observations_csv(
                WEATHER_PATH,
                weather_observations,
                WEATHER_COLUMNS,
                dedupe_columns=["source_record_id"],
            )

    feature_rows = 0
    if rebuild_features and not dry_run:
        feature_rows = rebuild_local_modeling_features(FEATURES_PATH)

    return LocalIngestionSummary(
        start_date=inferred_start_date.isoformat(),
        end_date=target_end_date.isoformat(),
        dry_run=dry_run,
        price_rows_fetched=price_rows,
        mix_rows_fetched=mix_rows,
        weather_rows_fetched=weather_rows,
        feature_rows_written=feature_rows,
    )


def repair_local_data(rebuild_features: bool = True) -> LocalIngestionSummary:
    """Repair local CSV timestamp fields without calling upstream APIs."""
    for path in [PRICE_PATH, MIX_PATH, WEATHER_PATH]:
        repair_existing_csv(path)
    feature_rows = rebuild_local_modeling_features(FEATURES_PATH) if rebuild_features else 0
    start = infer_refresh_start_date([PRICE_PATH, MIX_PATH, WEATHER_PATH]).isoformat()
    end_timestamp = max(
        timestamp for timestamp in [latest_timestamp(PRICE_PATH), latest_timestamp(MIX_PATH), latest_timestamp(WEATHER_PATH)]
        if timestamp is not None
    )
    return LocalIngestionSummary(
        start_date=start,
        end_date=end_timestamp.date().isoformat(),
        dry_run=False,
        feature_rows_written=feature_rows,
    )


def repair_existing_csv(path: Path) -> None:
    """Repair parseable timestamp fields in one existing processed CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return
    frame = pd.read_csv(path)
    if "timestamp_utc" not in frame:
        return
    frame["timestamp_utc"] = repair_missing_timestamps(frame)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    if "ingestion_timestamp_utc" in frame:
        frame["ingestion_timestamp_utc"] = repair_missing_ingestion_timestamps(frame)
        frame["ingestion_timestamp_utc"] = pd.to_datetime(
            frame["ingestion_timestamp_utc"],
            utc=True,
            errors="coerce",
        )
    sort_columns = [column for column in ["timestamp_utc", "scope", "region"] if column in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    normalize_timestamp_columns_for_csv(frame)
    frame.to_csv(path, index=False)


def infer_refresh_start_date(paths: list[Path]) -> date:
    """Infer a conservative refresh start date from local source CSVs."""
    latest_timestamps = [latest_timestamp(path) for path in paths]
    valid_timestamps = [timestamp for timestamp in latest_timestamps if timestamp is not None]
    if not valid_timestamps:
        return date(2023, 1, 1)
    return min(timestamp.date() for timestamp in valid_timestamps)


def resolve_refresh_start_date(
    target_end_date: date,
    explicit_start_date: date | None,
    lookback_days: int | None,
    source_paths: list[Path],
) -> date:
    """Resolve refresh start date from explicit, inferred, and bounded lookback inputs."""
    inferred_start_date = explicit_start_date or infer_refresh_start_date(source_paths)
    if lookback_days is not None:
        lookback_start = target_end_date - pd.Timedelta(days=max(lookback_days, 0))
        lookback_start_date = (
            lookback_start.date() if hasattr(lookback_start, "date") else lookback_start
        )
        inferred_start_date = max(inferred_start_date, lookback_start_date)
    return inferred_start_date


def latest_timestamp(path: Path) -> pd.Timestamp | None:
    """Return the latest timestamp in a local CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    frame = pd.read_csv(path, usecols=lambda column: column == "timestamp_utc")
    if frame.empty or "timestamp_utc" not in frame:
        return None
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.dropna().empty:
        return None
    return timestamps.max()


def upsert_observations_csv(
    path: Path,
    observations: list[Any],
    columns: list[str],
    dedupe_columns: list[str],
) -> int:
    """Merge observations into a local CSV and drop duplicate source records."""
    if not observations:
        return 0
    new_rows = pd.DataFrame([observation_to_row(observation, columns) for observation in observations])
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        frame = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        frame = new_rows

    for column in columns:
        if column not in frame:
            frame[column] = None
    if "timestamp_utc" in frame:
        frame["timestamp_utc"] = repair_missing_timestamps(frame)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    if "ingestion_timestamp_utc" in frame:
        frame["ingestion_timestamp_utc"] = pd.to_datetime(
            frame["ingestion_timestamp_utc"],
            utc=True,
            errors="coerce",
        )

    frame = frame.drop_duplicates(dedupe_columns, keep="last")
    sort_columns = [column for column in ["timestamp_utc", "scope", "region"] if column in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns)

    preferred_columns = [column for column in frame.columns if column not in columns]
    frame = frame[[*preferred_columns, *columns]]
    normalize_timestamp_columns_for_csv(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return len(new_rows)


def normalize_timestamp_columns_for_csv(frame: pd.DataFrame) -> None:
    """Write datetime columns as explicit ISO strings for stable CSV parsing."""
    for column in ["timestamp_utc", "ingestion_timestamp_utc", "created_at"]:
        if column not in frame:
            continue
        timestamps = pd.to_datetime(frame[column], utc=True, errors="coerce")
        frame[column] = timestamps.map(lambda value: value.isoformat() if pd.notna(value) else None)


def repair_missing_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Fill missing timestamp values from source_record_id when possible."""
    timestamps = frame["timestamp_utc"].copy()
    if "source_record_id" not in frame:
        return timestamps

    missing_mask = timestamps.isna() | (timestamps.astype(str).str.strip() == "")
    if not missing_mask.any():
        return timestamps

    repaired = frame.loc[missing_mask, "source_record_id"].astype(str).map(
        extract_timestamp_from_source_record_id
    )
    timestamps.loc[missing_mask] = repaired
    return timestamps


def repair_missing_ingestion_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Fill missing ingestion timestamps for repaired local rows."""
    timestamps = frame["ingestion_timestamp_utc"].copy()
    missing_mask = timestamps.isna() | (timestamps.astype(str).str.strip() == "")
    if missing_mask.any():
        timestamps.loc[missing_mask] = datetime.now(UTC).isoformat()
    return timestamps


def extract_timestamp_from_source_record_id(value: str) -> str | None:
    """Extract an ISO timestamp suffix from known source record IDs."""
    match = SOURCE_RECORD_TIMESTAMP_PATTERN.search(value)
    if not match:
        return None
    timestamp = match.group(1)
    return timestamp.replace("Z", "+00:00")


def observation_to_row(observation: Any, columns: list[str]) -> dict[str, Any]:
    """Convert a pydantic observation to a CSV-safe row."""
    values = observation.model_dump()
    return {column: to_csv_value(values.get(column)) for column in columns}


def to_csv_value(value: Any) -> Any:
    """Convert model values to CSV-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def rebuild_local_modeling_features(output_path: Path = FEATURES_PATH) -> int:
    """Rebuild modeling features from local processed source CSV files."""
    prices = pd.read_csv(PRICE_PATH, parse_dates=["timestamp_utc"])
    mix = pd.read_csv(MIX_PATH, parse_dates=["timestamp_utc"])
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp_utc"])

    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    mix["timestamp_utc"] = pd.to_datetime(mix["timestamp_utc"], utc=True)
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)

    price_frame = prices[
        (prices["region"] == "FR") & (prices["market"] == "day_ahead")
    ].drop_duplicates("timestamp_utc", keep="last")
    national_mix = mix[
        (mix["region"] == "FR") & (mix["scope"] == "national")
    ].drop_duplicates("timestamp_utc", keep="last")
    weather_agg = aggregate_weather(weather)

    base_frame = (
        price_frame[["timestamp_utc", "price_eur_mwh"]]
        .merge(national_mix[national_mix_columns()], on="timestamp_utc", how="inner")
        .merge(weather_agg, on="timestamp_utc", how="left")
        .sort_values("timestamp_utc")
    )
    features = build_price_modeling_features(base_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return int(len(features))


def national_mix_columns() -> list[str]:
    """Return mix columns required by the local feature builder."""
    return [
        "timestamp_utc",
        "consumption_mwh",
        "total_production_mwh",
        "nuclear_mwh",
        "thermal_mwh",
        "gas_mwh",
        "coal_mwh",
        "oil_mwh",
        "wind_mwh",
        "solar_mwh",
        "hydro_mwh",
        "bioenergy_mwh",
        "physical_exchanges_mwh",
    ]


def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regional weather observations into the feature-builder shape."""
    aggregations = {
        "temperature_c": ["mean", "min", "max"],
        "apparent_temperature_c": "mean",
        "wind_speed_mps": "mean",
        "wind_speed_80m_mps": "mean",
        "shortwave_radiation_wm2": "mean",
        "cloud_cover_pct": "mean",
        "precipitation_mm": ["mean", "sum"],
        "surface_pressure_hpa": "mean",
    }
    grouped = weather.groupby("timestamp_utc", as_index=False).agg(aggregations)
    grouped.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    return grouped.rename(
        columns={
            "temperature_c_mean": "avg_temperature_c",
            "temperature_c_min": "min_temperature_c",
            "temperature_c_max": "max_temperature_c",
            "apparent_temperature_c_mean": "avg_apparent_temperature_c",
            "wind_speed_mps_mean": "avg_wind_speed_mps",
            "wind_speed_80m_mps_mean": "avg_wind_speed_80m_mps",
            "shortwave_radiation_wm2_mean": "avg_shortwave_radiation_wm2",
            "cloud_cover_pct_mean": "avg_cloud_cover_pct",
            "precipitation_mm_mean": "avg_precipitation_mm",
            "precipitation_mm_sum": "total_precipitation_mm",
            "surface_pressure_hpa_mean": "avg_surface_pressure_hpa",
        }
    )


def parse_date(value: str | None) -> date | None:
    """Parse YYYY-MM-DD CLI values."""
    return date.fromisoformat(value) if value else None


def main(argv: list[str] | None = None) -> None:
    """Refresh local CSV data from upstream source APIs."""
    parser = argparse.ArgumentParser(description="Refresh local France energy CSV data.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Cap refresh start to a recent lookback window; useful for cloud scheduled runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-mix", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--repair-only", action="store_true")
    args = parser.parse_args(argv)

    if args.repair_only:
        summary = repair_local_data(rebuild_features=not args.skip_features)
    else:
        summary = refresh_local_data(
            start_date=parse_date(args.start_date),
            end_date=parse_date(args.end_date),
            lookback_days=args.lookback_days,
            dry_run=args.dry_run,
            include_prices=not args.skip_prices,
            include_mix=not args.skip_mix,
            include_weather=not args.skip_weather,
            rebuild_features=not args.skip_features,
            plan_only=args.plan_only,
        )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
