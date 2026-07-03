"""Supabase loading pipelines for France historical data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import httpx
from sqlalchemy.engine import Engine

from src.data.france_regions import RepresentativeLocation, get_regional_weather_locations
from src.data.load import (
    create_database_engine,
    upsert_electricity_prices,
    upsert_hourly_electricity_mix,
    upsert_weather_observations,
)
from src.data.source_config import FRANCE_END_DATE, FRANCE_START_DATE
from src.data.sources.energy_charts import (
    ENERGY_CHARTS_BASE_URL,
    ENERGY_CHARTS_MIN_INTERVAL_SECONDS,
    fetch_energy_charts_price_window,
    iter_energy_charts_price_windows,
)
from src.data.sources.odre import (
    ODRE_NATIONAL_HISTORICAL_DATASET,
    ODRE_REGIONAL_HISTORICAL_DATASET,
    aggregate_odre_to_hourly_mwh,
    fetch_odre_records,
)
from src.data.sources.open_meteo import fetch_open_meteo_weather

OPEN_METEO_MIN_INTERVAL_SECONDS = 0.2


@dataclass(frozen=True)
class LoadSummary:
    """Inserted/upserted row counts by canonical table."""

    electricity_price_rows: int = 0
    hourly_electricity_mix_rows: int = 0
    weather_rows: int = 0


def load_france_price_history(
    engine: Engine,
    start_date: date = FRANCE_START_DATE,
    end_date: date = FRANCE_END_DATE,
    batch_size: int = 1_000,
) -> int:
    """Fetch Energy-Charts France prices and upsert each window into Supabase."""
    total_rows = 0
    with httpx.Client(base_url=ENERGY_CHARTS_BASE_URL, timeout=60.0) as client:
        for window in iter_energy_charts_price_windows(start_date, end_date):
            observations = fetch_energy_charts_price_window(client=client, window=window)
            total_rows += upsert_electricity_prices(engine, observations, batch_size=batch_size)
            time.sleep(ENERGY_CHARTS_MIN_INTERVAL_SECONDS)
    return total_rows


def load_france_electricity_mix_history(
    engine: Engine,
    start_date: date = FRANCE_START_DATE,
    end_date: date = FRANCE_END_DATE,
    batch_size: int = 1_000,
) -> int:
    """Fetch ODRE electricity mix window-by-window and upsert into Supabase."""
    total_rows = 0
    from src.data.sources.odre import ODRE_WINDOW_DAYS
    from src.data.date_windows import iter_date_windows

    for window in iter_date_windows(start_date, end_date, ODRE_WINDOW_DAYS):
        national_records = fetch_odre_records(
            ODRE_NATIONAL_HISTORICAL_DATASET,
            window.start_date,
            window.end_date,
        )
        national_observations = aggregate_odre_to_hourly_mwh(national_records, scope="national")
        total_rows += upsert_hourly_electricity_mix(
            engine,
            national_observations,
            batch_size=batch_size,
        )

        regional_records = fetch_odre_records(
            ODRE_REGIONAL_HISTORICAL_DATASET,
            window.start_date,
            window.end_date,
        )
        regional_observations = aggregate_odre_to_hourly_mwh(regional_records, scope="regional")
        total_rows += upsert_hourly_electricity_mix(
            engine,
            regional_observations,
            batch_size=batch_size,
        )

    return total_rows


def load_france_weather_history(
    engine: Engine,
    start_date: date = FRANCE_START_DATE,
    end_date: date = FRANCE_END_DATE,
    locations: list[RepresentativeLocation] | None = None,
    batch_size: int = 1_000,
) -> int:
    """Fetch representative-city weather observations window-by-window."""
    weather_locations = locations or get_regional_weather_locations()
    total_rows = 0
    from src.data.sources.open_meteo import OPEN_METEO_WINDOW_DAYS
    from src.data.date_windows import iter_date_windows

    for location in weather_locations:
        for window in iter_date_windows(start_date, end_date, OPEN_METEO_WINDOW_DAYS):
            observations = fetch_open_meteo_weather(
                region=location.region,
                latitude=location.latitude,
                longitude=location.longitude,
                start_date=window.start_date,
                end_date=window.end_date,
            )
            total_rows += upsert_weather_observations(engine, observations, batch_size=batch_size)
            time.sleep(OPEN_METEO_MIN_INTERVAL_SECONDS)
    return total_rows


def load_france_history_to_supabase(
    database_url: str,
    start_date: date = FRANCE_START_DATE,
    end_date: date = FRANCE_END_DATE,
    include_prices: bool = True,
    include_electricity_mix: bool = True,
    include_weather: bool = True,
    batch_size: int = 1_000,
) -> LoadSummary:
    """Run the full France historical load into Supabase."""
    engine = create_database_engine(database_url)
    electricity_price_rows = (
        load_france_price_history(engine, start_date, end_date, batch_size)
        if include_prices
        else 0
    )
    hourly_electricity_mix_rows = (
        load_france_electricity_mix_history(engine, start_date, end_date, batch_size)
        if include_electricity_mix
        else 0
    )
    weather_rows = (
        load_france_weather_history(engine, start_date, end_date, batch_size=batch_size)
        if include_weather
        else 0
    )
    return LoadSummary(
        electricity_price_rows=electricity_price_rows,
        hourly_electricity_mix_rows=hourly_electricity_mix_rows,
        weather_rows=weather_rows,
    )
