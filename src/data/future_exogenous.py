"""Future exogenous data ingestion for operational clean-hour recommendations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.data.france_regions import get_regional_weather_locations
from src.data.sources.open_meteo import OPEN_METEO_HOURLY_FEATURES, WEATHER_FIELD_MAP

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data/processed/future_weather_forecast.csv"
OPEN_METEO_FORECAST_BASE_URL = "https://api.open-meteo.com"
OPEN_METEO_FORECAST_ENDPOINT = "/v1/forecast"


@dataclass(frozen=True)
class FutureExogenousSummary:
    """Summary of a future exogenous data refresh."""

    generated_at_utc: str
    horizon_hours: int
    weather_rows: int
    output_path: str


def ingest_future_exogenous(horizon_hours: int = 24, output_path: str | Path = OUTPUT_PATH) -> FutureExogenousSummary:
    """Fetch future weather forecasts for representative France regions."""
    rows: list[dict[str, Any]] = []
    with httpx.Client(base_url=OPEN_METEO_FORECAST_BASE_URL, timeout=60.0) as client:
        for location in get_regional_weather_locations():
            payload = fetch_open_meteo_forecast(
                client=client,
                latitude=location.latitude,
                longitude=location.longitude,
                horizon_hours=horizon_hours,
            )
            rows.extend(parse_open_meteo_forecast(location.region, payload, horizon_hours))

    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return FutureExogenousSummary(
        generated_at_utc=datetime.now(UTC).isoformat(),
        horizon_hours=horizon_hours,
        weather_rows=int(len(frame)),
        output_path=str(output.relative_to(ROOT)),
    )


def fetch_open_meteo_forecast(
    client: httpx.Client,
    latitude: float,
    longitude: float,
    horizon_hours: int,
) -> dict[str, Any]:
    """Fetch one location's hourly forecast payload."""
    response = client.get(
        OPEN_METEO_FORECAST_ENDPOINT,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(OPEN_METEO_HOURLY_FEATURES),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "forecast_hours": horizon_hours,
        },
    )
    response.raise_for_status()
    return response.json()


def parse_open_meteo_forecast(
    region: str,
    payload: dict[str, Any],
    horizon_hours: int,
) -> list[dict[str, Any]]:
    """Convert an Open-Meteo forecast payload to local weather rows."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])[:horizon_hours]
    rows: list[dict[str, Any]] = []
    generated_at = datetime.now(UTC).isoformat()
    for index, timestamp_value in enumerate(times):
        row: dict[str, Any] = {
            "source": "api",
            "source_record_id": f"open-meteo-forecast:{region}:{timestamp_value}+00:00",
            "region": region,
            "timestamp_utc": f"{timestamp_value}+00:00",
            "granularity": "1h",
            "ingestion_timestamp_utc": generated_at,
        }
        for source_field, target_field in WEATHER_FIELD_MAP.items():
            values = hourly.get(source_field)
            if isinstance(values, list) and index < len(values):
                row[target_field] = values[index]
        row["humidity_pct"] = row.get("relative_humidity_2m_pct")
        row["solar_irradiance_wm2"] = row.get("shortwave_radiation_wm2")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Fetch future exogenous weather data from the command line."""
    parser = argparse.ArgumentParser(description="Fetch future exogenous weather forecasts.")
    parser.add_argument("--horizon-hours", type=int, default=24)
    args = parser.parse_args(argv)
    summary = ingest_future_exogenous(horizon_hours=args.horizon_hours)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
