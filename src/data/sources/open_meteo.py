"""Open-Meteo historical weather extraction."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import httpx

from src.data.contracts import EnergySource, Granularity, WeatherObservation
from src.data.date_windows import iter_date_windows
from src.data.source_config import (
    OPEN_METEO_ARCHIVE_BASE_URL,
    OPEN_METEO_HISTORICAL_ENDPOINT,
    OPEN_METEO_HOURLY_FEATURES,
)

OPEN_METEO_WINDOW_DAYS = 31

WEATHER_FIELD_MAP = {
    "temperature_2m": "temperature_c",
    "apparent_temperature": "apparent_temperature_c",
    "relative_humidity_2m": "relative_humidity_2m_pct",
    "dew_point_2m": "dew_point_2m_c",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "cloud_cover": "cloud_cover_pct",
    "cloud_cover_low": "cloud_cover_low_pct",
    "cloud_cover_mid": "cloud_cover_mid_pct",
    "cloud_cover_high": "cloud_cover_high_pct",
    "shortwave_radiation": "shortwave_radiation_wm2",
    "direct_radiation": "direct_radiation_wm2",
    "diffuse_radiation": "diffuse_radiation_wm2",
    "wind_speed_10m": "wind_speed_mps",
    "wind_speed_80m": "wind_speed_80m_mps",
    "wind_direction_10m": "wind_direction_10m_deg",
    "wind_direction_80m": "wind_direction_80m_deg",
    "wind_gusts_10m": "wind_gusts_10m_mps",
    "surface_pressure": "surface_pressure_hpa",
    "weather_code": "weather_code",
}


def fetch_open_meteo_weather(
    region: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    base_url: str = OPEN_METEO_ARCHIVE_BASE_URL,
) -> list[WeatherObservation]:
    """Fetch hourly weather observations for one location."""
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        response = client.get(
            OPEN_METEO_HISTORICAL_ENDPOINT,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(OPEN_METEO_HOURLY_FEATURES),
                "timezone": "UTC",
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
            },
        )
        response.raise_for_status()
        return parse_open_meteo_weather(region=region, payload=response.json())


def parse_open_meteo_weather(region: str, payload: dict[str, Any]) -> list[WeatherObservation]:
    """Convert one Open-Meteo response into canonical weather observations."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    observations: list[WeatherObservation] = []

    for index, timestamp_value in enumerate(times):
        timestamp = datetime.fromisoformat(timestamp_value).replace(tzinfo=UTC)
        values: dict[str, Any] = {}
        for source_field, target_field in WEATHER_FIELD_MAP.items():
            series = hourly.get(source_field)
            if not isinstance(series, list) or index >= len(series):
                continue
            value = series[index]
            if value is None:
                continue
            values[target_field] = int(value) if target_field == "weather_code" else Decimal(str(value))

        observations.append(
            WeatherObservation(
                source=EnergySource.API,
                source_record_id=f"open-meteo:{region}:{timestamp.isoformat()}",
                region=region,
                timestamp_utc=timestamp,
                granularity=Granularity.HOURLY,
                humidity_pct=values.get("relative_humidity_2m_pct"),
                solar_irradiance_wm2=values.get("shortwave_radiation_wm2"),
                **values,
            )
        )

    return observations


def fetch_open_meteo_weather_windowed(
    region: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    base_url: str = OPEN_METEO_ARCHIVE_BASE_URL,
) -> list[WeatherObservation]:
    """Fetch hourly weather observations with explicit date-window splitting."""
    observations: list[WeatherObservation] = []
    for window in iter_date_windows(start_date, end_date, OPEN_METEO_WINDOW_DAYS):
        observations.extend(
            fetch_open_meteo_weather(
                region=region,
                latitude=latitude,
                longitude=longitude,
                start_date=window.start_date,
                end_date=window.end_date,
                base_url=base_url,
            )
        )
    return observations
