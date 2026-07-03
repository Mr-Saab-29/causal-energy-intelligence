"""ODRE éCO2mix extraction and hourly MWh aggregation."""

from __future__ import annotations

import time as sleep_timer
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import httpx

from src.data.contracts import EnergySource, Granularity, HourlyElectricityMixObservation
from src.data.date_windows import DateWindow, iter_date_windows
from src.data.source_config import (
    ODRE_BASE_URL,
    ODRE_NATIONAL_DATASET,
    ODRE_NATIONAL_HISTORICAL_DATASET,
    ODRE_REGIONAL_DATASET,
    ODRE_REGIONAL_HISTORICAL_DATASET,
)

ODRE_PAGE_LIMIT = 100
ODRE_WINDOW_DAYS = 7
ODRE_MIN_INTERVAL_SECONDS = 0.2
NATIONAL_REGION = "FR"

NATIONAL_MW_FIELDS = {
    "consommation": "consumption_mwh",
    "nucleaire": "nuclear_mwh",
    "gaz": "gas_mwh",
    "charbon": "coal_mwh",
    "fioul": "oil_mwh",
    "eolien": "wind_mwh",
    "eolien_terrestre": "onshore_wind_mwh",
    "eolien_offshore": "offshore_wind_mwh",
    "solaire": "solar_mwh",
    "hydraulique": "hydro_mwh",
    "pompage": "pumped_storage_mwh",
    "bioenergies": "bioenergy_mwh",
    "stockage_batterie": "battery_storage_mwh",
    "ech_physiques": "physical_exchanges_mwh",
}

REGIONAL_MW_FIELDS = {
    "consommation": "consumption_mwh",
    "thermique": "thermal_mwh",
    "nucleaire": "nuclear_mwh",
    "eolien": "wind_mwh",
    "solaire": "solar_mwh",
    "hydraulique": "hydro_mwh",
    "pompage": "pumped_storage_mwh",
    "bioenergies": "bioenergy_mwh",
    "stockage_batterie": "battery_storage_mwh",
    "destockage_batterie": "battery_storage_mwh",
    "ech_physiques": "physical_exchanges_mwh",
}


def fetch_odre_records(
    dataset: str,
    start_date: date,
    end_date: date,
    base_url: str = ODRE_BASE_URL,
) -> Iterator[dict[str, Any]]:
    """Fetch ODRE records using small date windows and offset pagination."""
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for window in iter_date_windows(start_date, end_date, ODRE_WINDOW_DAYS):
            offset = 0
            while True:
                response = client.get(
                    f"/api/explore/v2.1/catalog/datasets/{dataset}/records",
                    params={
                        "limit": ODRE_PAGE_LIMIT,
                        "offset": offset,
                        "order_by": "date_heure",
                        "where": _date_where_clause(window),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results", [])
                if not results:
                    break

                yield from results

                if len(results) < ODRE_PAGE_LIMIT:
                    break
                offset += ODRE_PAGE_LIMIT
                sleep_timer.sleep(ODRE_MIN_INTERVAL_SECONDS)


def aggregate_odre_to_hourly_mwh(
    records: Iterator[dict[str, Any]],
    scope: str,
) -> list[HourlyElectricityMixObservation]:
    """Convert ODRE quarter-hourly MW records into hourly MWh observations."""
    buckets: dict[tuple[str, datetime], dict[str, Decimal | str | datetime]] = defaultdict(dict)
    field_map = NATIONAL_MW_FIELDS if scope == "national" else REGIONAL_MW_FIELDS

    for record in records:
        timestamp = _parse_datetime(record["date_heure"])
        hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        region = NATIONAL_REGION if scope == "national" else str(record.get("libelle_region"))
        bucket_key = (region, hour_timestamp)
        bucket = buckets[bucket_key]
        bucket["region"] = region
        bucket["timestamp_utc"] = hour_timestamp
        bucket["source_record_id"] = f"odre:{scope}:{region}:{hour_timestamp.isoformat()}"

        for source_field, target_field in field_map.items():
            value = _to_decimal(record.get(source_field))
            if value is None:
                continue
            existing = bucket.get(target_field, Decimal("0"))
            bucket[target_field] = existing + value * Decimal("0.25")

        carbon_value = _to_decimal(record.get("taux_co2"))
        if carbon_value is not None:
            existing_carbon_sum = bucket.get("_carbon_sum", Decimal("0"))
            existing_carbon_count = bucket.get("_carbon_count", Decimal("0"))
            bucket["_carbon_sum"] = existing_carbon_sum + carbon_value
            bucket["_carbon_count"] = existing_carbon_count + Decimal("1")

    observations: list[HourlyElectricityMixObservation] = []
    production_fields = [
        "nuclear_mwh",
        "thermal_mwh",
        "gas_mwh",
        "coal_mwh",
        "oil_mwh",
        "wind_mwh",
        "solar_mwh",
        "hydro_mwh",
        "bioenergy_mwh",
    ]

    for bucket in buckets.values():
        carbon_count = bucket.pop("_carbon_count", Decimal("0"))
        carbon_sum = bucket.pop("_carbon_sum", Decimal("0"))
        if carbon_count:
            bucket["carbon_intensity_gco2_kwh"] = carbon_sum / carbon_count

        total_production = sum(
            bucket.get(field, Decimal("0")) for field in production_fields if bucket.get(field) is not None
        )
        bucket["total_production_mwh"] = total_production

        observations.append(
            HourlyElectricityMixObservation(
                source=EnergySource.API,
                scope=scope,
                granularity=Granularity.HOURLY,
                **bucket,
            )
        )

    return sorted(observations, key=lambda item: (item.region, item.timestamp_utc))


def fetch_france_national_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate national éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_NATIONAL_HISTORICAL_DATASET, start_date, end_date),
        scope="national",
    )


def fetch_france_regional_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate regional éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_REGIONAL_HISTORICAL_DATASET, start_date, end_date),
        scope="regional",
    )


def fetch_france_national_realtime_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate national real-time éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_NATIONAL_DATASET, start_date, end_date),
        scope="national",
    )


def fetch_france_regional_realtime_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate regional real-time éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_REGIONAL_DATASET, start_date, end_date),
        scope="regional",
    )


def _date_where_clause(window: DateWindow) -> str:
    start_timestamp = datetime.combine(window.start_date, time.min, tzinfo=UTC).isoformat()
    end_timestamp = datetime.combine(window.end_date + timedelta(days=1), time.min, tzinfo=UTC).isoformat()
    return (
        f'date_heure >= "{start_timestamp}" '
        f'and date_heure < "{end_timestamp}"'
    )


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _to_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    return Decimal(str(value))
