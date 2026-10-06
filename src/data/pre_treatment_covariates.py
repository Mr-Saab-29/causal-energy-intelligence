"""Ingest leakage-safe pre-treatment weather, market, and storage controls."""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import warnings
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from src.data.france_regions import get_regional_weather_locations
from src.data.http_retry import get_with_retries
from src.data.load import (
    create_database_engine,
    upsert_pre_treatment_hourly_covariates,
)
from src.data.sources.entsoe_causal import _optional_query

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/causal_pre_treatment.json"
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_pre_treatment_readiness.json"
COMPACT_TABLE = "causal_pre_treatment_hourly_covariates"
CORE_COLUMNS = (
    "weather_temperature_forecast_c_24h",
    "gas_price_usd_mmbtu_lag_2m",
    "coal_price_usd_mt_lag_2m",
    "eua_auction_price_eur_tco2",
    "hydro_storage_mwh_lag_1w",
)


def load_pre_treatment_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "contract_version",
        "history_start_date",
        "minimum_core_coverage",
        "weather",
        "fuel_prices",
        "carbon_price",
        "storage",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError("pre-treatment contract missing: " + ", ".join(missing))
    return contract


def fetch_weather_forecasts(
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract: dict[str, Any],
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Fetch 24-hour-ahead GFS temperature forecasts for France regions."""
    weather = contract["weather"]
    locations = get_regional_weather_locations()
    owns_client = client is None
    active_client = client or httpx.Client(timeout=120.0)
    records: list[dict[str, Any]] = []
    try:
        cursor = start.floor("D")
        while cursor < end:
            chunk_end = min(
                cursor + pd.Timedelta(days=int(weather["window_days"])),
                end.ceil("D"),
            )
            response = get_with_retries(
                active_client,
                weather["endpoint"],
                params={
                    "latitude": ",".join(str(item.latitude) for item in locations),
                    "longitude": ",".join(str(item.longitude) for item in locations),
                    "start_date": cursor.date().isoformat(),
                    "end_date": (chunk_end - pd.Timedelta(days=1)).date().isoformat(),
                    "hourly": weather["variable"],
                    "models": weather["model"],
                    "timezone": "UTC",
                },
                backoff_seconds=5.0,
            )
            payload = response.json()
            payloads = payload if isinstance(payload, list) else [payload]
            if len(payloads) != len(locations):
                raise RuntimeError(
                    "Open-Meteo returned an unexpected number of France locations"
                )
            for location, location_payload in zip(locations, payloads, strict=True):
                hourly = location_payload.get("hourly", {})
                times = hourly.get("time", [])
                values = hourly.get(weather["variable"], [])
                for timestamp, value in zip(times, values, strict=False):
                    if value is None:
                        continue
                    records.append(
                        {
                            "timestamp_utc": utc_boundary(timestamp),
                            "region": location.region,
                            "temperature_c": float(value),
                        }
                    )
            cursor = chunk_end
    finally:
        if owns_client:
            active_client.close()
    if not records:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "weather_temperature_forecast_c_24h",
                "weather_region_count",
            ]
        )
    frame = pd.DataFrame(records)
    return (
        frame.groupby("timestamp_utc", as_index=False)
        .agg(
            weather_temperature_forecast_c_24h=("temperature_c", "mean"),
            weather_region_count=("region", "nunique"),
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def fetch_world_bank_fuel_prices(
    contract: dict[str, Any],
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Download and parse monthly European TTF gas and coal regime controls."""
    fuel = contract["fuel_prices"]
    owns_client = client is None
    active_client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        landing = get_with_retries(
            active_client,
            fuel["landing_page"],
            backoff_seconds=5.0,
        ).text
        pattern = rf'href=["\']([^"\']*{re.escape(fuel["workbook_name"])})'
        match = re.search(pattern, html.unescape(landing), flags=re.IGNORECASE)
        if match is None:
            raise RuntimeError("World Bank monthly Pink Sheet link was not found")
        workbook_url = urljoin(fuel["landing_page"], match.group(1))
        workbook = get_with_retries(
            active_client,
            workbook_url,
            backoff_seconds=5.0,
        ).content
    finally:
        if owns_client:
            active_client.close()
    return parse_world_bank_fuel_workbook(workbook, contract)


def parse_world_bank_fuel_workbook(
    content: bytes,
    contract: dict[str, Any],
) -> pd.DataFrame:
    fuel = contract["fuel_prices"]
    frame = pd.read_excel(
        io.BytesIO(content),
        sheet_name="Monthly Prices",
        header=4,
        skiprows=[5],
    )
    date_column = frame.columns[0]
    required = [fuel["gas_column"], fuel["coal_column"]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError("World Bank fuel columns missing: " + ", ".join(missing))
    output = pd.DataFrame(
        {
            "reference_period": frame[date_column].astype(str),
            "gas_price_usd_mmbtu": pd.to_numeric(
                frame[fuel["gas_column"]], errors="coerce"
            ),
            "coal_price_usd_mt": pd.to_numeric(
                frame[fuel["coal_column"]], errors="coerce"
            ),
        }
    )
    output = output[output["reference_period"].str.fullmatch(r"\d{4}M\d{2}")]
    period_values = output.pop("reference_period").str.replace(
        r"^(\d{4})M(\d{2})$", r"\1-\2", regex=True
    )
    output["reference_month"] = pd.PeriodIndex(period_values, freq="M")
    return output.dropna(subset=["gas_price_usd_mmbtu", "coal_price_usd_mt"])


def fetch_eex_carbon_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract: dict[str, Any],
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Fetch public EEX ETS1 primary-auction prices and publication timestamps."""
    carbon = contract["carbon_price"]
    owns_client = client is None
    active_client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    frames: list[pd.DataFrame] = []
    try:
        archive = get_with_retries(
            active_client,
            carbon["history_archive"],
            backoff_seconds=5.0,
        ).content
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for name in bundle.namelist():
                year_match = re.search(r"(20\d{2})-data\.xlsx$", name)
                if year_match is None:
                    continue
                year = int(year_match.group(1))
                if start.year - 1 <= year <= end.year:
                    frames.append(parse_eex_auction_workbook(bundle.read(name), contract))
        current_url = carbon["current_workbook_template"].format(year=end.year)
        current = get_with_retries(
            active_client,
            current_url,
            backoff_seconds=5.0,
        ).content
        frames.append(parse_eex_auction_workbook(current, contract))
    finally:
        if owns_client:
            active_client.close()
    if not frames:
        return pd.DataFrame(columns=["available_at_utc", "eua_auction_price_eur_tco2"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("available_at_utc", keep="last")
        .sort_values("available_at_utc")
        .reset_index(drop=True)
    )


def parse_eex_auction_workbook(
    content: bytes,
    contract: dict[str, Any],
) -> pd.DataFrame:
    carbon = contract["carbon_price"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        frame = pd.read_excel(io.BytesIO(content), sheet_name=0, header=5)
    required = {
        "Time",
        "Contract",
        "Status",
        "Auction Price €/tCO2",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("EEX auction workbook columns changed")
    eligible = (
        frame["Status"].astype(str).str.lower().eq("successful")
        & frame["Contract"].astype(str).eq(carbon["contract"])
    )
    if "Zone" in frame.columns:
        eligible &= frame["Zone"].astype(str).eq(carbon["zone"])
    elif "Auction Name" in frame.columns:
        eligible &= frame["Auction Name"].astype(str).str.contains(
            rf"\b{re.escape(carbon['zone'])}$", regex=True
        )
    else:
        raise RuntimeError("EEX auction workbook has no zone identifier")
    frame = frame[eligible].copy()
    published_local = pd.to_datetime(frame["Time"], errors="coerce")
    if published_local.dt.tz is None:
        published_local = published_local.dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward"
        )
    frame["available_at_utc"] = published_local.dt.tz_convert("UTC")
    frame["eua_auction_price_eur_tco2"] = pd.to_numeric(
        frame["Auction Price €/tCO2"], errors="coerce"
    )
    return frame[
        ["available_at_utc", "eua_auction_price_eur_tco2"]
    ].dropna()


def fetch_hydro_storage(
    api_token: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract: dict[str, Any],
    *,
    client: EntsoePandasClient | None = None,
) -> pd.DataFrame:
    """Fetch France aggregate hydro storage in bounded yearly windows."""
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN is required")
    storage = contract["storage"]
    entsoe = client or EntsoePandasClient(api_key=api_token)
    series: list[pd.Series] = []
    cursor = start - pd.Timedelta(days=21)
    query_end = end + pd.Timedelta(days=1)
    while cursor < query_end:
        chunk_end = min(cursor + pd.DateOffset(years=1), query_end)
        values = _optional_query(
            entsoe.query_aggregate_water_reservoirs_and_hydro_storage,
            dataset="aggregate_water_reservoirs_and_hydro_storage:FR",
            country_code=storage["area"],
            start=cursor,
            end=chunk_end,
        )
        if values is not None:
            item = values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values
            series.append(item)
        cursor = chunk_end
    if not series:
        return pd.DataFrame(columns=["observed_at_utc", "hydro_storage_mwh"])
    combined = pd.concat(series)
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined = pd.to_numeric(combined, errors="coerce")
    combined = combined[combined.ge(0)].sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return pd.DataFrame(
        {
            "observed_at_utc": combined.index,
            "hydro_storage_mwh": combined.to_numpy(dtype=float),
        }
    )


def assemble_hourly_covariates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract: dict[str, Any],
    weather: pd.DataFrame,
    fuel: pd.DataFrame,
    carbon: pd.DataFrame,
    storage: pd.DataFrame,
    *,
    compacted_at: datetime | None = None,
) -> pd.DataFrame:
    """Align each source to the latest value available before each decision hour."""
    frame = pd.DataFrame(
        {"timestamp_utc": pd.date_range(start, end, freq="h", inclusive="left")}
    )
    frame = frame.merge(weather, on="timestamp_utc", how="left")

    lag_months = int(contract["fuel_prices"]["publication_lag_months"])
    timestamp_month = frame["timestamp_utc"].dt.tz_localize(None).dt.to_period("M")
    frame["fuel_reference_period"] = timestamp_month - lag_months
    frame = frame.merge(
        fuel,
        left_on="fuel_reference_period",
        right_on="reference_month",
        how="left",
    )
    frame["gas_price_usd_mmbtu_lag_2m"] = frame.pop("gas_price_usd_mmbtu")
    frame["coal_price_usd_mt_lag_2m"] = frame.pop("coal_price_usd_mt")
    frame["fuel_reference_month"] = frame["reference_month"].dt.to_timestamp()

    carbon_values = carbon.rename(
        columns={"available_at_utc": "carbon_auction_at_utc"}
    ).sort_values("carbon_auction_at_utc")
    frame = pd.merge_asof(
        frame.sort_values("timestamp_utc"),
        carbon_values,
        left_on="timestamp_utc",
        right_on="carbon_auction_at_utc",
        direction="backward",
        allow_exact_matches=False,
    )

    lag_days = int(contract["storage"]["availability_lag_days"])
    storage_values = storage.copy()
    storage_values["storage_available_at_utc"] = (
        storage_values["observed_at_utc"] + pd.Timedelta(days=lag_days)
    )
    storage_values = storage_values.rename(
        columns={
            "observed_at_utc": "hydro_storage_observed_at_utc",
            "hydro_storage_mwh": "hydro_storage_mwh_lag_1w",
        }
    ).sort_values("storage_available_at_utc")
    frame = pd.merge_asof(
        frame.sort_values("timestamp_utc"),
        storage_values,
        left_on="timestamp_utc",
        right_on="storage_available_at_utc",
        direction="backward",
        allow_exact_matches=False,
    )
    frame["vintage_quality"] = "historical_final"
    frame["compacted_at_utc"] = compacted_at or datetime.now(UTC)
    frame["weather_region_count"] = frame["weather_region_count"].fillna(0).astype(int)
    return frame[
        [
            "timestamp_utc",
            "vintage_quality",
            "weather_temperature_forecast_c_24h",
            "weather_region_count",
            "gas_price_usd_mmbtu_lag_2m",
            "coal_price_usd_mt_lag_2m",
            "eua_auction_price_eur_tco2",
            "hydro_storage_mwh_lag_1w",
            "fuel_reference_month",
            "carbon_auction_at_utc",
            "hydro_storage_observed_at_utc",
            "compacted_at_utc",
        ]
    ]


def build_pre_treatment_readiness(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("pre-treatment covariates are empty")
    threshold = float(contract["minimum_core_coverage"])
    coverage = {
        column: round(float(frame[column].notna().mean()), 4)
        for column in CORE_COLUMNS
    }
    critical = [
        f"low_coverage:{column}:{coverage[column]:.4f}"
        for column in CORE_COLUMNS
        if coverage[column] < threshold
    ]
    fuel_period = frame["timestamp_utc"].dt.tz_localize(None).dt.to_period("M") - int(
        contract["fuel_prices"]["publication_lag_months"]
    )
    actual_fuel_period = pd.to_datetime(frame["fuel_reference_month"]).dt.to_period("M")
    temporal_violations = {
        "fuel_reference_after_safe_month": int((actual_fuel_period > fuel_period).sum()),
        "carbon_auction_not_before_hour": int(
            (frame["carbon_auction_at_utc"] >= frame["timestamp_utc"]).fillna(False).sum()
        ),
        "storage_not_lagged_one_week": int(
            (
                frame["hydro_storage_observed_at_utc"]
                + pd.Timedelta(days=int(contract["storage"]["availability_lag_days"]))
                >= frame["timestamp_utc"]
            )
            .fillna(False)
            .sum()
        ),
    }
    for name, count in temporal_violations.items():
        if count:
            critical.append(f"temporal_violation:{name}:{count}")
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contract_version": contract["contract_version"],
        "status": "fail" if critical else "pass",
        "pre_treatment_covariates_ready": not critical,
        "window": {
            "start_utc": frame["timestamp_utc"].min().isoformat(),
            "end_utc": frame["timestamp_utc"].max().isoformat(),
            "rows": int(len(frame)),
        },
        "coverage": coverage,
        "critical_issues": critical,
        "temporal_violations": temporal_violations,
        "source_quality": {
            "weather": "archived_gfs_fixed_24h_lead",
            "fuel_prices": "world_bank_monthly_ttf_and_australian_coal_lagged_two_months",
            "carbon_price": "last_completed_eu_primary_auction",
            "storage": "weekly_historical_state_lagged_one_week",
        },
        "known_limitations": contract.get("known_limitations", []),
    }


def validate_pre_treatment_schema(engine: Engine) -> None:
    if COMPACT_TABLE not in set(inspect(engine).get_table_names(schema="public")):
        raise RuntimeError(
            "Pre-treatment migration is incomplete. Apply db/causal_grid_compact.sql."
        )


def ingest_pre_treatment_covariates(
    api_token: str,
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Fetch, causally align, validate, and upsert one bounded hourly window."""
    validate_pre_treatment_schema(engine)
    contract = load_pre_treatment_contract(config_path)
    start = utc_boundary(start)
    end = utc_boundary(end)
    if end <= start:
        raise ValueError("end must be after start")
    weather = fetch_weather_forecasts(start, end, contract)
    fuel = fetch_world_bank_fuel_prices(contract)
    carbon = fetch_eex_carbon_prices(start, end, contract)
    storage = fetch_hydro_storage(api_token, start, end, contract)
    frame = assemble_hourly_covariates(
        start,
        end,
        contract,
        weather,
        fuel,
        carbon,
        storage,
    )
    report = build_pre_treatment_readiness(frame, contract)
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not report["pre_treatment_covariates_ready"]:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(
            "pre-treatment quality failed: " + ", ".join(report["critical_issues"])
        )
    rows = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    report["upserted_rows"] = upsert_pre_treatment_hourly_covariates(engine, rows)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def utc_boundary(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").floor("h")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ingest leakage-safe causal pre-treatment covariates."
    )
    parser.add_argument("--api-token", default=os.environ.get("ENTSOE_API_TOKEN"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--lookback-days", type=int, default=31)
    parser.add_argument("--historical-backfill", action="store_true")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)
    contract = load_pre_treatment_contract(args.config_path)
    today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC")
    if args.historical_backfill:
        start = utc_boundary(contract["history_start_date"])
    elif args.start_date:
        start = utc_boundary(args.start_date)
    else:
        start = today - timedelta(days=args.lookback_days)
    end = (
        utc_boundary(args.end_date) + timedelta(days=1)
        if args.end_date
        else today
    )
    missing = [
        name
        for name, value in {
            "ENTSOE_API_TOKEN": args.api_token,
            "DATABASE_URL": args.database_url,
        }.items()
        if not value
    ]
    if missing:
        parser.error("missing required configuration: " + ", ".join(missing))
    report = ingest_pre_treatment_covariates(
        args.api_token,
        create_database_engine(args.database_url),
        start,
        end,
        config_path=args.config_path,
        report_path=args.report_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["window"]["rows"],
                "upserted_rows": report["upserted_rows"],
                "report": args.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
