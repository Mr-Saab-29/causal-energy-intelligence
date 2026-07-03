"""ENTSO-E Transparency Platform day-ahead price extraction."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import httpx

from src.data.contracts import ElectricityPriceObservation, EnergySource, Granularity
from src.data.date_windows import iter_date_windows
from src.data.source_config import (
    ENTSOE_BASE_URL,
    ENTSOE_DAY_AHEAD_PRICES_DOCUMENT_TYPE,
    ENTSOE_ENDPOINT,
    ENTSOE_MARKET_AGREEMENT_TYPE,
    FRANCE_ENTSOE_BIDDING_ZONE,
)

ENTSOE_WINDOW_DAYS = 365


def fetch_entsoe_day_ahead_prices(
    api_token: str,
    start_date: date,
    end_date: date,
    bidding_zone: str = FRANCE_ENTSOE_BIDDING_ZONE,
    base_url: str = ENTSOE_BASE_URL,
) -> list[ElectricityPriceObservation]:
    """Fetch day-ahead prices using ENTSO-E's documented one-year request bound."""
    observations: list[ElectricityPriceObservation] = []
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for window in iter_date_windows(start_date, end_date, ENTSOE_WINDOW_DAYS):
            response = client.get(
                ENTSOE_ENDPOINT,
                params={
                    "securityToken": api_token,
                    "documentType": ENTSOE_DAY_AHEAD_PRICES_DOCUMENT_TYPE,
                    "in_Domain": bidding_zone,
                    "out_Domain": bidding_zone,
                    "contract_MarketAgreement.type": ENTSOE_MARKET_AGREEMENT_TYPE,
                    "periodStart": _entsoe_timestamp(window.start_date),
                    "periodEnd": _entsoe_timestamp(window.end_date + timedelta(days=1)),
                },
            )
            response.raise_for_status()
            observations.extend(parse_entsoe_day_ahead_prices(response.text, bidding_zone))
    return observations


def parse_entsoe_day_ahead_prices(xml_text: str, bidding_zone: str) -> list[ElectricityPriceObservation]:
    """Parse ENTSO-E XML day-ahead price response into hourly observations."""
    root = ET.fromstring(xml_text)
    namespace = _namespace(root)
    observations: list[ElectricityPriceObservation] = []

    for period in root.findall(f".//{namespace}Period"):
        start_node = period.find(f"{namespace}timeInterval/{namespace}start")
        resolution_node = period.find(f"{namespace}resolution")
        if start_node is None or not start_node.text:
            continue

        period_start = datetime.fromisoformat(start_node.text.replace("Z", "+00:00")).astimezone(UTC)
        resolution = resolution_node.text if resolution_node is not None else "PT60M"

        for point in period.findall(f"{namespace}Point"):
            position_node = point.find(f"{namespace}position")
            price_node = point.find(f"{namespace}price.amount")
            if position_node is None or price_node is None or not position_node.text or not price_node.text:
                continue

            position = int(position_node.text)
            timestamp = period_start + _resolution_delta(resolution) * (position - 1)
            observations.append(
                ElectricityPriceObservation(
                    source=EnergySource.API,
                    source_record_id=f"entsoe:{bidding_zone}:{timestamp.isoformat()}",
                    region="FR",
                    timestamp_utc=timestamp,
                    granularity=Granularity.HOURLY,
                    market="day_ahead",
                    price_eur_mwh=Decimal(price_node.text),
                    currency="EUR",
                )
            )

    return observations


def iter_entsoe_price_windows(start_date: date, end_date: date) -> Iterator[tuple[str, str]]:
    """Yield ENTSO-E request timestamps for inspection and tests."""
    for window in iter_date_windows(start_date, end_date, ENTSOE_WINDOW_DAYS):
        yield _entsoe_timestamp(window.start_date), _entsoe_timestamp(window.end_date + timedelta(days=1))


def _entsoe_timestamp(value: date) -> str:
    return value.strftime("%Y%m%d0000")


def _namespace(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag.split("}")[0] + "}"
    return ""


def _resolution_delta(resolution: str) -> timedelta:
    if resolution == "PT15M":
        return timedelta(minutes=15)
    if resolution == "PT30M":
        return timedelta(minutes=30)
    return timedelta(hours=1)

