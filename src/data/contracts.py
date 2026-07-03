"""Canonical data contracts for the energy intelligence platform."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EnergySource(StrEnum):
    """Supported upstream source categories."""

    API = "api"
    CSV = "csv"
    MANUAL = "manual"


class Granularity(StrEnum):
    """Supported observation granularities."""

    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    HOURLY = "1h"
    DAILY = "1d"


class BaseObservation(BaseModel):
    """Shared fields for all time-indexed observations."""

    model_config = ConfigDict(extra="forbid")

    source: EnergySource
    source_record_id: str | None = None
    region: str = Field(min_length=1, examples=["FR", "DE", "US-CAISO"])
    timestamp_utc: datetime
    granularity: Granularity
    ingestion_timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("timestamp_utc", "ingestion_timestamp_utc")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require timezone-aware datetimes to avoid ambiguous time joins."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("datetime fields must be timezone-aware")
        return value


class ElectricityPriceObservation(BaseObservation):
    """Market electricity price observation."""

    market: str = Field(min_length=1, examples=["day_ahead", "real_time"])
    price_eur_mwh: Decimal
    currency: str = Field(default="EUR", min_length=3, max_length=3)


class CarbonIntensityObservation(BaseObservation):
    """Grid carbon intensity observation."""

    carbon_intensity_gco2_kwh: Decimal = Field(ge=0)
    estimation_method: str | None = Field(default=None, examples=["measured", "estimated"])


class GridDemandObservation(BaseObservation):
    """Electricity demand/load observation."""

    demand_mw: Decimal = Field(ge=0)
    demand_type: str = Field(default="actual", examples=["actual", "forecast"])


class RenewableGenerationObservation(BaseObservation):
    """Renewable generation observation."""

    generation_mw: Decimal = Field(ge=0)
    technology: str = Field(min_length=1, examples=["wind", "solar", "hydro"])


class HourlyElectricityMixObservation(BaseObservation):
    """Hourly MWh electricity production and consumption observation."""

    scope: str = Field(examples=["national", "regional"])
    consumption_mwh: Decimal | None = Field(default=None, ge=0)
    total_production_mwh: Decimal | None = Field(default=None, ge=0)
    nuclear_mwh: Decimal | None = Field(default=None, ge=0)
    thermal_mwh: Decimal | None = Field(default=None, ge=0)
    gas_mwh: Decimal | None = Field(default=None, ge=0)
    coal_mwh: Decimal | None = Field(default=None, ge=0)
    oil_mwh: Decimal | None = Field(default=None, ge=0)
    wind_mwh: Decimal | None = Field(default=None, ge=0)
    onshore_wind_mwh: Decimal | None = Field(default=None, ge=0)
    offshore_wind_mwh: Decimal | None = Field(default=None, ge=0)
    solar_mwh: Decimal | None = Field(default=None, ge=0)
    hydro_mwh: Decimal | None = Field(default=None, ge=0)
    pumped_storage_mwh: Decimal | None = None
    bioenergy_mwh: Decimal | None = Field(default=None, ge=0)
    battery_storage_mwh: Decimal | None = None
    physical_exchanges_mwh: Decimal | None = None
    carbon_intensity_gco2_kwh: Decimal | None = Field(default=None, ge=0)


class WeatherObservation(BaseObservation):
    """Weather covariates used for forecasting and causal adjustment."""

    temperature_c: Decimal | None = None
    apparent_temperature_c: Decimal | None = None
    relative_humidity_2m_pct: Decimal | None = Field(default=None, ge=0, le=100)
    dew_point_2m_c: Decimal | None = None
    precipitation_mm: Decimal | None = Field(default=None, ge=0)
    rain_mm: Decimal | None = Field(default=None, ge=0)
    snowfall_cm: Decimal | None = Field(default=None, ge=0)
    cloud_cover_pct: Decimal | None = Field(default=None, ge=0, le=100)
    cloud_cover_low_pct: Decimal | None = Field(default=None, ge=0, le=100)
    cloud_cover_mid_pct: Decimal | None = Field(default=None, ge=0, le=100)
    cloud_cover_high_pct: Decimal | None = Field(default=None, ge=0, le=100)
    shortwave_radiation_wm2: Decimal | None = Field(default=None, ge=0)
    direct_radiation_wm2: Decimal | None = Field(default=None, ge=0)
    diffuse_radiation_wm2: Decimal | None = Field(default=None, ge=0)
    wind_speed_mps: Decimal | None = Field(default=None, ge=0)
    wind_speed_80m_mps: Decimal | None = Field(default=None, ge=0)
    wind_direction_10m_deg: Decimal | None = Field(default=None, ge=0, le=360)
    wind_direction_80m_deg: Decimal | None = Field(default=None, ge=0, le=360)
    wind_gusts_10m_mps: Decimal | None = Field(default=None, ge=0)
    surface_pressure_hpa: Decimal | None = Field(default=None, ge=0)
    weather_code: int | None = None
    solar_irradiance_wm2: Decimal | None = Field(default=None, ge=0)
    humidity_pct: Decimal | None = Field(default=None, ge=0, le=100)


class WorkloadWindow(BaseModel):
    """Schedulable workload window for what-if optimization."""

    model_config = ConfigDict(extra="forbid")

    workload_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    earliest_start_utc: datetime
    latest_end_utc: datetime
    duration_minutes: int = Field(gt=0)
    power_kw: Decimal = Field(gt=0)
    max_delay_minutes: int = Field(ge=0)
    service_level: str = Field(default="standard")

    @field_validator("earliest_start_utc", "latest_end_utc")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require timezone-aware datetimes to avoid ambiguous schedules."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("datetime fields must be timezone-aware")
        return value


class ApiPageResult(BaseModel):
    """Normalized result from one paginated upstream API request."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    endpoint: str
    page_token: str | None = None
    next_page_token: str | None = None
    offset: int | None = None
    limit: int | None = None
    records: list[dict[str, object]]
    raw_response: dict[str, object] | list[object] | None = None
