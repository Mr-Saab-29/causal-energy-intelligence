"""Energy-Charts day-ahead price extraction."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import httpx

from src.data.contracts import ElectricityPriceObservation, EnergySource, Granularity
from src.data.date_windows import DateWindow, iter_calendar_year_windows
from src.data.source_config import (
    ENERGY_CHARTS_BASE_URL,
    ENERGY_CHARTS_FRANCE_BIDDING_ZONE,
    ENERGY_CHARTS_PRICE_ENDPOINT,
)

ENERGY_CHARTS_WINDOW_DAYS = 365
ENERGY_CHARTS_MIN_INTERVAL_SECONDS = 10.0
ENERGY_CHARTS_MAX_RETRIES = 5
ENERGY_CHARTS_BACKOFF_SECONDS = 60.0


def fetch_energy_charts_day_ahead_prices(
    start_date: date,
    end_date: date,
    bidding_zone: str = ENERGY_CHARTS_FRANCE_BIDDING_ZONE,
    base_url: str = ENERGY_CHARTS_BASE_URL,
) -> list[ElectricityPriceObservation]:
    """Fetch Energy-Charts day-ahead prices for a bidding zone."""
    observations: list[ElectricityPriceObservation] = []
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for window in iter_energy_charts_price_windows(start_date, end_date):
            observations.extend(
                fetch_energy_charts_price_window(
                    client=client,
                    window=window,
                    bidding_zone=bidding_zone,
                )
            )
            time.sleep(ENERGY_CHARTS_MIN_INTERVAL_SECONDS)

    return _deduplicate_price_observations(observations)


def iter_energy_charts_price_windows(start_date: date, end_date: date) -> list[DateWindow]:
    """Return calendar-year windows for Energy-Charts price requests."""
    return list(iter_calendar_year_windows(start_date, end_date))


def fetch_energy_charts_price_window(
    client: httpx.Client,
    window: DateWindow,
    bidding_zone: str = ENERGY_CHARTS_FRANCE_BIDDING_ZONE,
) -> list[ElectricityPriceObservation]:
    """Fetch one Energy-Charts window with explicit retry/backoff for rate limits."""
    params = {
        "bzn": bidding_zone,
        "start": window.start_date.isoformat(),
        "end": window.end_date.isoformat(),
    }
    for attempt in range(ENERGY_CHARTS_MAX_RETRIES + 1):
        response = client.get(ENERGY_CHARTS_PRICE_ENDPOINT, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            return parse_energy_charts_day_ahead_prices(
                payload=response.json(),
                bidding_zone=bidding_zone,
            )

        retry_after = response.headers.get("retry-after")
        if retry_after:
            wait_seconds = float(retry_after)
        else:
            wait_seconds = ENERGY_CHARTS_BACKOFF_SECONDS * (attempt + 1)

        if attempt == ENERGY_CHARTS_MAX_RETRIES:
            response.raise_for_status()

        time.sleep(wait_seconds)

    return []


def parse_energy_charts_day_ahead_prices(
    payload: dict[str, Any],
    bidding_zone: str = ENERGY_CHARTS_FRANCE_BIDDING_ZONE,
) -> list[ElectricityPriceObservation]:
    """Convert Energy-Charts price payload into canonical price observations."""
    timestamps = payload.get("unix_seconds", [])
    prices = payload.get("price", [])
    unit = payload.get("unit")

    if unit != "EUR / MWh":
        raise ValueError(f"Unsupported Energy-Charts price unit: {unit}")
    if not isinstance(timestamps, list) or not isinstance(prices, list):
        raise ValueError("Energy-Charts payload must contain unix_seconds and price arrays")
    if len(timestamps) != len(prices):
        raise ValueError("Energy-Charts timestamp and price arrays have different lengths")

    observations: list[ElectricityPriceObservation] = []
    for timestamp_value, price_value in zip(timestamps, prices, strict=True):
        if price_value is None:
            continue
        timestamp = datetime.fromtimestamp(int(timestamp_value), tz=UTC)
        observations.append(
            ElectricityPriceObservation(
                source=EnergySource.API,
                source_record_id=f"energy-charts:{bidding_zone}:{timestamp.isoformat()}",
                region="FR",
                timestamp_utc=timestamp,
                granularity=Granularity.HOURLY,
                market="day_ahead",
                price_eur_mwh=Decimal(str(price_value)),
                currency="EUR",
            )
        )

    return observations


def _deduplicate_price_observations(
    observations: list[ElectricityPriceObservation],
) -> list[ElectricityPriceObservation]:
    deduplicated = {observation.source_record_id: observation for observation in observations}
    return sorted(deduplicated.values(), key=lambda observation: observation.timestamp_utc)
