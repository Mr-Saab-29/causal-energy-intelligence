"""Database loading utilities for transformed energy records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.data.contracts import (
    ElectricityPriceObservation,
    HourlyElectricityMixObservation,
    WeatherObservation,
)

PRICE_COLUMNS = [
    "source",
    "source_record_id",
    "region",
    "timestamp_utc",
    "granularity",
    "market",
    "price_eur_mwh",
    "currency",
    "ingestion_timestamp_utc",
]

HOURLY_ELECTRICITY_MIX_COLUMNS = [
    "source",
    "source_record_id",
    "region",
    "scope",
    "timestamp_utc",
    "granularity",
    "consumption_mwh",
    "total_production_mwh",
    "nuclear_mwh",
    "thermal_mwh",
    "gas_mwh",
    "coal_mwh",
    "oil_mwh",
    "wind_mwh",
    "onshore_wind_mwh",
    "offshore_wind_mwh",
    "solar_mwh",
    "hydro_mwh",
    "pumped_storage_mwh",
    "bioenergy_mwh",
    "battery_storage_mwh",
    "physical_exchanges_mwh",
    "carbon_intensity_gco2_kwh",
    "ingestion_timestamp_utc",
]

WEATHER_COLUMNS = [
    "source",
    "source_record_id",
    "region",
    "timestamp_utc",
    "granularity",
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_2m_pct",
    "dew_point_2m_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    "cloud_cover_pct",
    "cloud_cover_low_pct",
    "cloud_cover_mid_pct",
    "cloud_cover_high_pct",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "wind_speed_mps",
    "wind_speed_80m_mps",
    "wind_direction_10m_deg",
    "wind_direction_80m_deg",
    "wind_gusts_10m_mps",
    "surface_pressure_hpa",
    "weather_code",
    "solar_irradiance_wm2",
    "humidity_pct",
    "ingestion_timestamp_utc",
]

CONFLICT_COLUMNS = ["source", "source_record_id"]


def load_energy_data(records: list[dict[str, object]]) -> int:
    """Load transformed records into the configured warehouse."""
    return len(records)


def create_database_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for Supabase's Postgres connection string."""
    return create_engine(database_url, pool_pre_ping=True)


def insert_raw_api_page(
    engine: Engine,
    source_name: str,
    endpoint: str,
    request_params: dict[str, object],
    response_payload: dict[str, object] | list[object] | None,
) -> None:
    """Persist one raw API response page for lineage and reprocessing."""
    statement = text(
        """
        insert into raw_api_pages (
            source_name,
            endpoint,
            request_params,
            response_payload
        )
        values (
            :source_name,
            :endpoint,
            cast(:request_params as jsonb),
            cast(:response_payload as jsonb)
        )
        """
    )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "source_name": source_name,
                "endpoint": endpoint,
                "request_params": json.dumps(request_params, default=str),
                "response_payload": json.dumps(response_payload, default=str),
            },
        )


def upsert_electricity_prices(
    engine: Engine,
    observations: Sequence[ElectricityPriceObservation],
    batch_size: int = 1_000,
) -> int:
    """Upsert canonical electricity price observations into Supabase."""
    return _upsert_observations(
        engine=engine,
        table_name="electricity_prices",
        columns=PRICE_COLUMNS,
        observations=observations,
        batch_size=batch_size,
    )


def upsert_hourly_electricity_mix(
    engine: Engine,
    observations: Sequence[HourlyElectricityMixObservation],
    batch_size: int = 1_000,
) -> int:
    """Upsert hourly MWh electricity mix observations into Supabase."""
    return _upsert_observations(
        engine=engine,
        table_name="hourly_electricity_mix",
        columns=HOURLY_ELECTRICITY_MIX_COLUMNS,
        observations=observations,
        batch_size=batch_size,
    )


def upsert_weather_observations(
    engine: Engine,
    observations: Sequence[WeatherObservation],
    batch_size: int = 1_000,
) -> int:
    """Upsert canonical weather observations into Supabase."""
    return _upsert_observations(
        engine=engine,
        table_name="weather_observations",
        columns=WEATHER_COLUMNS,
        observations=observations,
        batch_size=batch_size,
    )


def build_upsert_sql(table_name: str, columns: Sequence[str]) -> str:
    """Build a safe Postgres upsert statement for whitelisted identifiers."""
    _validate_identifier(table_name)
    for column in columns:
        _validate_identifier(column)

    insert_columns = ", ".join(columns)
    value_columns = ", ".join(f":{column}" for column in columns)
    conflict_columns = ", ".join(CONFLICT_COLUMNS)
    update_columns = [
        column
        for column in columns
        if column not in {*CONFLICT_COLUMNS, "created_at", "id"}
    ]
    update_assignments = ", ".join(
        f"{column} = excluded.{column}" for column in update_columns
    )
    return (
        f"insert into {table_name} ({insert_columns}) "
        f"values ({value_columns}) "
        f"on conflict ({conflict_columns}) do update set {update_assignments}"
    )


def _upsert_observations(
    engine: Engine,
    table_name: str,
    columns: Sequence[str],
    observations: Sequence[Any],
    batch_size: int,
) -> int:
    if not observations:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    statement = text(build_upsert_sql(table_name, columns))
    rows = [_observation_to_row(observation, columns) for observation in observations]

    with engine.begin() as connection:
        for batch in _batched(rows, batch_size):
            connection.execute(statement, batch)

    return len(rows)


def _observation_to_row(observation: Any, columns: Sequence[str]) -> dict[str, Any]:
    values = observation.model_dump()
    if not values.get("source_record_id"):
        raise ValueError("source_record_id is required for idempotent Supabase upserts")

    return {
        column: _to_database_value(values.get(column))
        for column in columns
    }


def _to_database_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value
    return value


def _batched(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start_index in range(0, len(rows), batch_size):
        yield list(rows[start_index : start_index + batch_size])


def _validate_identifier(identifier: str) -> None:
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
