"""ENTSO-E causal-grid extraction and canonical normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from requests import HTTPError

from src.data.contracts import (
    BalancingObservation,
    CrossBorderObservation,
    EnergySource,
    GenerationOutageObservation,
    Granularity,
    GridForecastObservation,
)
from src.data.source_config import (
    FRANCE_DIRECT_ENTSOE_NEIGHBORS,
    FRANCE_ENTSOE_AREA,
)

VINTAGE_OPERATIONAL = "operational_snapshot"
VINTAGE_HISTORICAL = "historical_final"
ACTIVATED_RESERVE_TYPES = {
    "A96": "afrr",
    "A97": "mfrr",
    "A98": "replacement_reserve",
}


@dataclass(frozen=True)
class EntsoeCausalData:
    """Canonical records returned by one bounded ENTSO-E extraction."""

    cross_border: list[CrossBorderObservation]
    forecasts: list[GridForecastObservation]
    outages: list[GenerationOutageObservation]
    balancing: list[BalancingObservation]
    warnings: tuple[str, ...] = ()


def fetch_entsoe_causal_data(
    api_token: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    historical_backfill: bool = False,
    snapshot_at: datetime | None = None,
    client: EntsoePandasClient | None = None,
) -> EntsoeCausalData:
    """Fetch the agreed France causal-grid datasets for one bounded window."""
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN is required")
    start = _utc_timestamp(start)
    end = _utc_timestamp(end)
    if end <= start:
        raise ValueError("end must be after start")
    snapshot_timestamp = pd.Timestamp(snapshot_at or datetime.now(UTC))
    if snapshot_timestamp.tzinfo is None:
        raise ValueError("snapshot_at must be timezone-aware")
    snapshot = snapshot_timestamp.tz_convert("UTC").floor("h").to_pydatetime()
    vintage = VINTAGE_HISTORICAL if historical_backfill else VINTAGE_OPERATIONAL
    entsoe = client or EntsoePandasClient(api_key=api_token)
    warnings: list[str] = []

    cross_border: list[CrossBorderObservation] = []
    for neighbor in FRANCE_DIRECT_ENTSOE_NEIGHBORS:
        for source_area, target_area in (
            (FRANCE_ENTSOE_AREA, neighbor),
            (neighbor, FRANCE_ENTSOE_AREA),
        ):
            queries = (
                ("physical_flow", entsoe.query_crossborder_flows),
                (
                    "scheduled_exchange",
                    lambda **kwargs: entsoe.query_scheduled_exchanges(
                        **kwargs,
                        dayahead=True,
                    ),
                ),
                (
                    "day_ahead_capacity",
                    entsoe.query_net_transfer_capacity_dayahead,
                ),
            )
            for metric, query in queries:
                values = _optional_query(
                    query,
                    country_code_from=source_area,
                    country_code_to=target_area,
                    start=start,
                    end=end,
                )
                cross_border.extend(
                    normalize_cross_border_series(
                        values,
                        source_area,
                        target_area,
                        metric,
                        snapshot,
                        vintage,
                    )
                )

    forecasts: list[GridForecastObservation] = []
    load = _optional_query(
        entsoe.query_load_forecast,
        country_code=FRANCE_ENTSOE_AREA,
        start=start,
        end=end,
    )
    forecasts.extend(
        normalize_forecast_frame(load, FRANCE_ENTSOE_AREA, snapshot, vintage, "load")
    )
    renewable = _optional_query(
        entsoe.query_wind_and_solar_forecast,
        country_code=FRANCE_ENTSOE_AREA,
        start=start,
        end=end,
    )
    forecasts.extend(
        normalize_forecast_frame(
            renewable,
            FRANCE_ENTSOE_AREA,
            snapshot,
            vintage,
        )
    )

    outage_frame = _optional_query(
        entsoe.query_unavailability_of_generation_units,
        country_code=FRANCE_ENTSOE_AREA,
        start=start,
        end=end,
    )
    outages = normalize_outage_frame(
        outage_frame,
        FRANCE_ENTSOE_AREA,
        snapshot,
        vintage,
    )

    balancing: list[BalancingObservation] = []
    realized_end = min(end, pd.Timestamp(snapshot))
    if start >= realized_end:
        return EntsoeCausalData(cross_border, forecasts, outages, balancing)
    balancing_queries = (
        ("imbalance_price", "EUR/MWh", entsoe.query_imbalance_prices),
        ("imbalance_volume", "MWh", entsoe.query_imbalance_volumes),
        (
            "activated_energy_price",
            "EUR/MWh",
            entsoe.query_activated_balancing_energy_prices,
        ),
    )
    for metric, unit, query in balancing_queries:
        frame = _optional_query(
            query,
            country_code=FRANCE_ENTSOE_AREA,
            start=start,
            end=realized_end,
        )
        balancing.extend(
            normalize_balancing_frame(
                frame,
                FRANCE_ENTSOE_AREA,
                metric,
                unit,
                snapshot,
                vintage,
            )
        )
    for business_type, reserve_type in ACTIVATED_RESERVE_TYPES.items():
        frame = _optional_query(
            entsoe.query_activated_balancing_energy,
            country_code=FRANCE_ENTSOE_AREA,
            start=start,
            end=realized_end,
            business_type=business_type,
            dataset=f"activated_energy:{reserve_type}",
            tolerate_bad_request=True,
            warnings=warnings,
        )
        balancing.extend(
            normalize_balancing_frame(
                frame,
                FRANCE_ENTSOE_AREA,
                "activated_energy",
                "MW",
                snapshot,
                vintage,
                reserve_type=reserve_type,
            )
        )

    return EntsoeCausalData(
        cross_border,
        forecasts,
        outages,
        balancing,
        tuple(warnings),
    )


def normalize_cross_border_series(
    values: pd.Series | pd.DataFrame | None,
    source_area: str,
    target_area: str,
    metric: str,
    snapshot_at: datetime,
    vintage_quality: str,
) -> list[CrossBorderObservation]:
    """Normalize one directional ENTSO-E cross-border series."""
    series = _first_numeric_series(values)
    if series is None:
        return []
    granularity = _infer_granularity(series.index)
    rows: list[CrossBorderObservation] = []
    for timestamp, value in series.items():
        number = _decimal_or_none(value)
        if number is None or number < 0:
            continue
        timestamp_utc = _utc_datetime(timestamp)
        if (
            vintage_quality == VINTAGE_OPERATIONAL
            and metric != "physical_flow"
            and timestamp_utc < snapshot_at
        ):
            continue
        snapshot_suffix = (
            f":{snapshot_at.isoformat()}"
            if vintage_quality == VINTAGE_OPERATIONAL and metric != "physical_flow"
            else ""
        )
        rows.append(
            CrossBorderObservation(
                source=EnergySource.API,
                source_record_id=(
                    f"entsoe:{metric}:{source_area}:{target_area}:"
                    f"{timestamp_utc.isoformat()}{snapshot_suffix}"
                ),
                region=FRANCE_ENTSOE_AREA,
                timestamp_utc=timestamp_utc,
                granularity=granularity,
                from_bidding_zone=source_area,
                to_bidding_zone=target_area,
                metric=metric,
                value_mw=number,
                snapshot_at_utc=snapshot_at,
                vintage_quality=vintage_quality,
            )
        )
    return rows


def normalize_forecast_frame(
    values: pd.Series | pd.DataFrame | None,
    area: str,
    snapshot_at: datetime,
    vintage_quality: str,
    forced_type: str | None = None,
) -> list[GridForecastObservation]:
    """Normalize load, wind, and solar forecasts while retaining ingestion vintage."""
    frame = _numeric_frame(values)
    if frame.empty:
        return []
    granularity = _infer_granularity(frame.index)
    rows: list[GridForecastObservation] = []
    for column in frame.columns:
        forecast_type = forced_type or _forecast_type(column)
        if forecast_type is None:
            continue
        for timestamp, value in frame[column].items():
            number = _decimal_or_none(value)
            if number is None or number < 0:
                continue
            timestamp_utc = _utc_datetime(timestamp)
            if (
                vintage_quality == VINTAGE_OPERATIONAL
                and timestamp_utc < snapshot_at
            ):
                continue
            generated_at = (
                snapshot_at if vintage_quality == VINTAGE_OPERATIONAL else None
            )
            horizon = (
                max(0, int((timestamp_utc - generated_at).total_seconds() // 3600))
                if generated_at is not None
                else None
            )
            snapshot_key = (
                snapshot_at.isoformat()
                if vintage_quality == VINTAGE_OPERATIONAL
                else VINTAGE_HISTORICAL
            )
            rows.append(
                GridForecastObservation(
                    source=EnergySource.API,
                    source_record_id=(
                        f"entsoe:forecast:{area}:{forecast_type}:"
                        f"{timestamp_utc.isoformat()}:{snapshot_key}"
                    ),
                    region=area,
                    timestamp_utc=timestamp_utc,
                    granularity=granularity,
                    forecast_type=forecast_type,
                    forecast_mw=number,
                    snapshot_at_utc=snapshot_at,
                    forecast_generated_at_utc=generated_at,
                    forecast_horizon_hours=horizon,
                    vintage_quality=vintage_quality,
                )
            )
    return rows


def normalize_outage_frame(
    values: pd.DataFrame | None,
    area: str,
    snapshot_at: datetime,
    vintage_quality: str,
) -> list[GenerationOutageObservation]:
    """Normalize ENTSO-E A80 unit-unavailability rows and preserve revisions."""
    if values is None or values.empty:
        return []
    frame = values.reset_index()
    rows: list[GenerationOutageObservation] = []
    for record in frame.to_dict(orient="records"):
        start = _datetime_or_none(record.get("start"))
        end = _datetime_or_none(record.get("end"))
        mrid = _text_or_none(record.get("mrid"))
        if start is None or end is None or mrid is None or end <= start:
            continue
        revision = int(record.get("revision") or 0)
        available = _decimal_or_none(record.get("avail_qty"))
        nominal = _decimal_or_none(record.get("nominal_power"))
        unavailable = (
            max(Decimal("0"), nominal - available)
            if nominal is not None and available is not None
            else None
        )
        business_type = _text_or_none(record.get("businesstype"))
        outage_type = {
            "A53": "planned",
            "A54": "unplanned",
            "Planned maintenance": "planned",
            "Unplanned outage": "unplanned",
        }.get(business_type, "other")
        publication = _datetime_or_none(record.get("created_doc_time"))
        rows.append(
            GenerationOutageObservation(
                source=EnergySource.API,
                source_record_id=(
                    f"entsoe:outage:{mrid}:{revision}:{vintage_quality}"
                ),
                region=area,
                timestamp_utc=start,
                granularity=_granularity_from_resolution(record.get("resolution")),
                outage_mrid=mrid,
                revision_number=revision,
                outage_type=outage_type,
                status=_text_or_none(record.get("docstatus")),
                end_utc=end,
                production_resource_id=_text_or_none(
                    record.get("production_resource_id")
                ),
                production_resource_name=_text_or_none(
                    record.get("production_resource_name")
                ),
                production_type=_text_or_none(record.get("plant_type")),
                nominal_capacity_mw=nominal,
                available_capacity_mw=available,
                unavailable_capacity_mw=unavailable,
                publication_timestamp_utc=publication,
                snapshot_at_utc=snapshot_at,
                vintage_quality=vintage_quality,
            )
        )
    return rows


def normalize_balancing_frame(
    values: pd.Series | pd.DataFrame | None,
    area: str,
    metric: str,
    unit: str,
    snapshot_at: datetime,
    vintage_quality: str,
    *,
    reserve_type: str | None = None,
) -> list[BalancingObservation]:
    """Melt an ENTSO-E balancing result while retaining native resolution."""
    frame = _numeric_frame(values)
    if frame.empty:
        return []
    granularity = _infer_granularity(frame.index)
    rows: list[BalancingObservation] = []
    for column in frame.columns:
        label = _column_label(column)
        direction = _direction(label)
        for timestamp, value in frame[column].items():
            number = _decimal_or_none(value)
            if number is None:
                continue
            timestamp_utc = _utc_datetime(timestamp)
            rows.append(
                BalancingObservation(
                    source=EnergySource.API,
                    source_record_id=(
                        f"entsoe:{metric}:{area}:{label}:"
                        f"{timestamp_utc.isoformat()}"
                    ),
                    region=area,
                    timestamp_utc=timestamp_utc,
                    granularity=granularity,
                    metric=metric,
                    value=number,
                    unit=unit,
                    direction=direction,
                    reserve_type=reserve_type or _reserve_type(label),
                    snapshot_at_utc=snapshot_at,
                    vintage_quality=vintage_quality,
                )
            )
    return rows


def _optional_query(
    query: Any,
    *,
    dataset: str | None = None,
    tolerate_bad_request: bool = False,
    warnings: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    try:
        return query(**kwargs)
    except NoMatchingDataError:
        return None
    except HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        label = dataset or getattr(query, "__name__", "unknown_dataset")
        if tolerate_bad_request and status == 400:
            if warnings is not None:
                warnings.append(
                    f"{label} unavailable from the legacy ENTSO-E endpoint (HTTP 400)"
                )
            return None
        raise RuntimeError(
            f"ENTSO-E query {label} failed with HTTP {status or 'unknown'}"
        ) from None


def _numeric_frame(values: pd.Series | pd.DataFrame | None) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame()
    frame = values.to_frame(name="value") if isinstance(values, pd.Series) else values.copy()
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[frame.index.notna()]
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.dropna(axis=1, how="all")


def _first_numeric_series(values: pd.Series | pd.DataFrame | None) -> pd.Series | None:
    frame = _numeric_frame(values)
    return frame.iloc[:, 0] if not frame.empty and len(frame.columns) else None


def _infer_granularity(index: pd.Index) -> Granularity:
    timestamps = pd.DatetimeIndex(index).dropna().sort_values().unique()
    if len(timestamps) < 2:
        return Granularity.HOURLY
    minutes = int(np.median(np.diff(timestamps.asi8)) // 60_000_000_000)
    if minutes <= 5:
        return Granularity.FIVE_MINUTES
    if minutes <= 15:
        return Granularity.FIFTEEN_MINUTES
    if minutes <= 30:
        return Granularity.THIRTY_MINUTES
    return Granularity.HOURLY


def _granularity_from_resolution(value: Any) -> Granularity:
    return {
        "PT5M": Granularity.FIVE_MINUTES,
        "PT15M": Granularity.FIFTEEN_MINUTES,
        "PT30M": Granularity.THIRTY_MINUTES,
        "PT60M": Granularity.HOURLY,
    }.get(str(value), Granularity.HOURLY)


def _forecast_type(column: Any) -> str | None:
    label = _column_label(column).lower()
    if "solar" in label:
        return "solar"
    if "offshore" in label:
        return "wind_offshore"
    if "wind" in label or "onshore" in label:
        return "wind_onshore"
    if "load" in label:
        return "load"
    return None


def _direction(label: str) -> str | None:
    lowered = label.lower()
    if "down" in lowered or "short" in lowered:
        return "down"
    if "up" in lowered or "long" in lowered:
        return "up"
    return None


def _reserve_type(label: str) -> str | None:
    lowered = label.lower()
    for token in ("afrr", "mfrr", "fcr", "replacement", "rr"):
        if token in lowered:
            return token
    return None


def _column_label(column: Any) -> str:
    parts = column if isinstance(column, tuple) else (column,)
    return "|".join(str(part) for part in parts if str(part) not in {"", "nan"})


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return Decimal(str(number))


def _datetime_or_none(value: Any) -> datetime | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(timestamp) else timestamp.to_pydatetime()


def _text_or_none(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _utc_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
