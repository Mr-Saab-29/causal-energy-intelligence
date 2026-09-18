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
from src.data.http_retry import get_with_retries
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
                response = get_with_retries(
                    client,
                    f"/api/explore/v2.1/catalog/datasets/{dataset}/records",
                    params={
                        "limit": ODRE_PAGE_LIMIT,
                        "offset": offset,
                        "order_by": "date_heure",
                        "where": _date_where_clause(window),
                    },
                    max_retries=5,
                    backoff_seconds=10.0,
                )
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
    *,
    dataset: str,
) -> list[HourlyElectricityMixObservation]:
    """Integrate complete source intervals; leave incomplete hourly fields missing.

    Historical actuals are half-hourly even though their records include empty
    quarter-hour slots for the separately published consumption forecasts.
    Never infer cadence from the number of surviving observations.
    """
    cadences = {
        ODRE_NATIONAL_HISTORICAL_DATASET: ("national", 30),
        ODRE_REGIONAL_HISTORICAL_DATASET: ("regional", 30),
        ODRE_NATIONAL_DATASET: ("national", 15),
        ODRE_REGIONAL_DATASET: ("regional", 15),
    }
    if dataset not in cadences or cadences[dataset][0] != scope:
        raise ValueError(f"Unknown ODRE dataset/scope: {dataset}/{scope}")
    minutes = cadences[dataset][1]
    weight = Decimal(minutes) / Decimal(60)
    expected_slots = set(range(0, 60, minutes))
    buckets: dict[tuple[str, datetime], dict[str, Decimal | str | datetime]] = defaultdict(dict)
    slots = defaultdict(lambda: defaultdict(set))
    seen = {}
    field_map = NATIONAL_MW_FIELDS if scope == "national" else REGIONAL_MW_FIELDS

    for record in records:
        timestamp = _parse_datetime(record["date_heure"])
        hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        region = NATIONAL_REGION if scope == "national" else str(record.get("libelle_region"))
        bucket_key = (region, hour_timestamp)
        record_key = (region, timestamp)
        # DST source rows can repeat a UTC instant with different local labels
        # or unused forecasts. Count matching actuals once; never choose between
        # conflicting measurements or merge complementary partial records.
        signature = (record.get("nature"), *(
            _to_decimal(record.get(field)) for field in [*field_map, "taux_co2"]
        ))
        if record_key in seen:
            if seen[record_key] != signature:
                raise ValueError(f"Conflicting duplicate ODRE observation: {record_key}")
            continue
        seen[record_key] = signature
        nature = record.get("nature")
        if nature in {"Données consolidées", "Données définitives", "Données temps réel"}:
            nature_minutes = 15 if nature == "Données temps réel" else 30
            if nature_minutes != minutes:
                raise ValueError(f"ODRE nature contradicts dataset cadence: {dataset}/{nature}")
        bucket = buckets[bucket_key]
        bucket["region"] = region
        bucket["timestamp_utc"] = hour_timestamp
        bucket["source_record_id"] = f"odre:{scope}:{region}:{hour_timestamp.isoformat()}"

        for source_field, target_field in field_map.items():
            value = _to_decimal(record.get(source_field))
            if value is None:
                continue
            if not value.is_finite() or timestamp.minute not in expected_slots or timestamp.second or timestamp.microsecond:
                raise ValueError(f"Invalid ODRE value/interval: {source_field} at {timestamp}")
            slots[bucket_key][target_field].add(timestamp.minute)
            existing = bucket.get(target_field, Decimal("0"))
            bucket[target_field] = existing + value * weight

        carbon_value = _to_decimal(record.get("taux_co2"))
        if carbon_value is not None:
            if not carbon_value.is_finite() or timestamp.minute not in expected_slots or timestamp.second or timestamp.microsecond:
                raise ValueError(f"Invalid ODRE carbon interval at {timestamp}")
            slots[bucket_key]["carbon_intensity_gco2_kwh"].add(timestamp.minute)
            existing_carbon_sum = bucket.get("_carbon_sum", Decimal("0"))
            existing_carbon_count = bucket.get("_carbon_count", Decimal("0"))
            bucket["_carbon_sum"] = existing_carbon_sum + carbon_value
            bucket["_carbon_count"] = existing_carbon_count + Decimal("1")

    observations: list[HourlyElectricityMixObservation] = []
    production_fields = [
        "nuclear_mwh",
        "gas_mwh",
        "coal_mwh",
        "oil_mwh",
        "wind_mwh",
        "solar_mwh",
        "hydro_mwh",
        "bioenergy_mwh",
    ]
    if scope == "regional":
        production_fields = ["thermal_mwh", "nuclear_mwh", "wind_mwh", "solar_mwh", "hydro_mwh", "bioenergy_mwh"]

    for bucket_key, bucket in buckets.items():
        for field in set(field_map.values()):
            if slots[bucket_key][field] != expected_slots:
                bucket.pop(field, None)
        carbon_count = bucket.pop("_carbon_count", Decimal("0"))
        carbon_sum = bucket.pop("_carbon_sum", Decimal("0"))
        if carbon_count and slots[bucket_key]["carbon_intensity_gco2_kwh"] == expected_slots:
            bucket["carbon_intensity_gco2_kwh"] = carbon_sum / carbon_count

        total_production = (
            sum(bucket[field] for field in production_fields)
            if all(field in bucket for field in production_fields) else None
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
        dataset=ODRE_NATIONAL_HISTORICAL_DATASET,
    )


def fetch_france_regional_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate regional éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_REGIONAL_HISTORICAL_DATASET, start_date, end_date),
        scope="regional",
        dataset=ODRE_REGIONAL_HISTORICAL_DATASET,
    )


def fetch_france_national_realtime_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate national real-time éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_NATIONAL_DATASET, start_date, end_date),
        scope="national",
        dataset=ODRE_NATIONAL_DATASET,
    )


def fetch_france_regional_realtime_hourly_mix(start_date: date, end_date: date) -> list[HourlyElectricityMixObservation]:
    """Fetch and aggregate regional real-time éCO2mix data for France."""
    return aggregate_odre_to_hourly_mwh(
        fetch_odre_records(ODRE_REGIONAL_DATASET, start_date, end_date),
        scope="regional",
        dataset=ODRE_REGIONAL_DATASET,
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
