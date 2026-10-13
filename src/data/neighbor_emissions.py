"""Ingest compact hourly emissions for France's directly connected zones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from src.carbon.intensity import load_emission_factor_config
from src.data.causal_archive_compact import SupabaseStorage
from src.data.entsoe_causal_ingest import build_window
from src.data.load import create_database_engine, upsert_neighbor_hourly_emissions
from src.data.sources.entsoe_causal import VINTAGE_HISTORICAL, VINTAGE_OPERATIONAL, _optional_query

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/neighbor_emissions.json"
DEFAULT_FACTORS_PATH = ROOT / "config/emission_factors.yaml"
DEFAULT_ARCHIVE_ROOT = ROOT / "data/archive/entsoe-neighbor-emissions"
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/neighbor_emissions_readiness.json"
COMPACT_TABLE = "causal_neighbor_hourly_emissions"
ELEXON_GENERATION_URL = (
    "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH"
)


def load_neighbor_emissions_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "contract_version",
        "areas",
        "area_sources",
        "elexon_fuel_type_to_production_type",
        "minimum_hourly_coverage",
        "minimum_named_factor_coverage",
        "fallback_factor",
        "production_type_to_factor_category",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("neighbor emissions contract missing: " + ", ".join(missing))
    return config


def fetch_neighbor_generation(
    api_token: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    areas: tuple[str, ...],
    client: EntsoePandasClient | None = None,
    elexon_client: httpx.Client | None = None,
    area_sources: dict[str, str] | None = None,
    elexon_fuel_type_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch and normalize actual generation for every required neighboring zone."""
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN is required")
    entsoe = client or EntsoePandasClient(api_key=api_token)
    sources = area_sources or {area: "entsoe" for area in areas}
    frames: list[pd.DataFrame] = []
    for area in areas:
        source = sources.get(area)
        if source == "elexon":
            normalized = fetch_elexon_generation(
                start,
                end,
                area=area,
                client=elexon_client,
                fuel_type_mapping=elexon_fuel_type_mapping,
            )
        elif source == "entsoe":
            values = _optional_query(
                entsoe.query_generation,
                dataset=f"actual_generation_per_type:{area}",
                country_code=area,
                start=start,
                end=end,
                nett=False,
            )
            normalized = normalize_generation_frame(values, area)
        else:
            raise ValueError(f"unsupported neighbor generation source for {area}: {source}")
        if normalized.empty:
            raise RuntimeError(f"{source} returned no generation-by-type data for {area}")
        normalized["source_system"] = source
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True)


def fetch_elexon_generation(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    area: str = "GB",
    client: httpx.Client | None = None,
    fuel_type_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch Great Britain half-hourly generation by fuel from Elexon FUELHH."""
    mapping = fuel_type_mapping or load_neighbor_emissions_contract()[
        "elexon_fuel_type_to_production_type"
    ]
    owns_client = client is None
    active_client = client or httpx.Client(timeout=60.0)
    payload_rows: list[dict[str, Any]] = []
    try:
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=7), end)
            response = active_client.get(
                ELEXON_GENERATION_URL,
                params={
                    "publishDateTimeFrom": chunk_start.isoformat(),
                    # Elexon treats both bounds as inclusive and rejects ranges
                    # exceeding seven days.
                    "publishDateTimeTo": chunk_end.isoformat(),
                },
            )
            if not response.is_success:
                raise RuntimeError(
                    f"Elexon generation query failed with HTTP {response.status_code}"
                )
            payload_rows.extend(response.json().get("data", []))
            chunk_start = chunk_end
    finally:
        if owns_client:
            active_client.close()
    records: list[dict[str, Any]] = []
    unknown_fuel_types: set[str] = set()
    for item in payload_rows:
        timestamp = pd.Timestamp(item.get("startTime"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        timestamp = timestamp.tz_convert("UTC")
        if timestamp < start or timestamp >= end:
            continue
        fuel_type = str(item.get("fuelType"))
        if fuel_type.startswith("INT"):
            continue
        production_type = mapping.get(fuel_type)
        if production_type is None:
            unknown_fuel_types.add(fuel_type)
            continue
        quantity = pd.to_numeric(item.get("generation"), errors="coerce")
        if pd.isna(quantity) or float(quantity) < 0:
            continue
        records.append(
            {
                "timestamp_utc": timestamp,
                "publish_time": pd.Timestamp(item.get("publishTime")),
                "bidding_zone": area,
                "production_type": production_type,
                "generation_mw": float(quantity),
            }
        )
    if unknown_fuel_types:
        raise ValueError(
            "unmapped Elexon fuel types: " + ", ".join(sorted(unknown_fuel_types))
        )
    if not records:
        return empty_generation_frame()
    raw = pd.DataFrame(records)
    raw["publish_time"] = pd.to_datetime(raw["publish_time"], utc=True)
    raw = raw.sort_values("publish_time").drop_duplicates(
        ["timestamp_utc", "bidding_zone", "production_type"], keep="last"
    )
    raw["hour"] = raw["timestamp_utc"].dt.floor("h")
    return (
        raw.groupby(["hour", "bidding_zone", "production_type"], as_index=False)
        .agg(
            generation_mwh=("generation_mw", "mean"),
            source_interval_count=("generation_mw", "count"),
        )
        .rename(columns={"hour": "timestamp_utc"})
        .sort_values(["timestamp_utc", "bidding_zone", "production_type"])
        .reset_index(drop=True)
    )


def normalize_generation_frame(
    values: pd.Series | pd.DataFrame | None,
    area: str,
) -> pd.DataFrame:
    """Convert ENTSO-E generation columns and native intervals into hourly MWh."""
    if values is None:
        return empty_generation_frame()
    frame = values.to_frame() if isinstance(values, pd.Series) else values.copy()
    if frame.empty:
        return empty_generation_frame()
    frame.index = pd.to_datetime(frame.index, utc=True)
    records: list[pd.DataFrame] = []
    for column in frame.columns:
        production_type, metric = generation_column_parts(column)
        if metric != "Actual Aggregated":
            continue
        values_mw = pd.to_numeric(frame[column], errors="coerce").dropna()
        values_mw = values_mw[values_mw >= 0]
        if values_mw.empty:
            continue
        hourly = values_mw.resample("h").agg(["mean", "count"])
        hourly = hourly[hourly["count"] > 0]
        records.append(
            pd.DataFrame(
                {
                    "timestamp_utc": hourly.index,
                    "bidding_zone": area,
                    "production_type": production_type,
                    # Mean MW over one hour is numerically equal to energy in MWh.
                    "generation_mwh": hourly["mean"].to_numpy(),
                    "source_interval_count": hourly["count"].astype(int).to_numpy(),
                }
            )
        )
    if not records:
        return empty_generation_frame()
    output = pd.concat(records, ignore_index=True)
    return (
        output.groupby(
            ["timestamp_utc", "bidding_zone", "production_type"],
            as_index=False,
        )
        .agg(
            generation_mwh=("generation_mwh", "sum"),
            source_interval_count=("source_interval_count", "sum"),
        )
        .sort_values(["timestamp_utc", "bidding_zone", "production_type"])
        .reset_index(drop=True)
    )


def generation_column_parts(column: object) -> tuple[str, str]:
    if isinstance(column, tuple):
        values = [str(value) for value in column if value not in (None, "")]
        production_type = values[0] if values else "Other"
        metric = values[1] if len(values) > 1 else "Actual Aggregated"
        return production_type, metric
    return str(column), "Actual Aggregated"


def empty_generation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp_utc",
            "bidding_zone",
            "production_type",
            "generation_mwh",
            "source_interval_count",
            "source_system",
        ]
    )


def build_neighbor_hourly_emissions(
    generation: pd.DataFrame,
    contract: dict[str, Any],
    *,
    vintage_quality: str,
    emission_factors: dict[str, float] | None = None,
    compacted_at: datetime | None = None,
) -> pd.DataFrame:
    """Apply the versioned factor map and preserve fallback uncertainty bounds."""
    if generation.empty:
        return pd.DataFrame()
    factors = emission_factors or load_emission_factor_config(DEFAULT_FACTORS_PATH)[
        "direct_operational_emissions"
    ]
    factors = dict(factors) | {"zero_direct": 0.0}
    factor_categories = contract["production_type_to_factor_category"]
    fallback = contract["fallback_factor"]
    frame = generation.copy()
    frame["factor_category"] = frame["production_type"].map(factor_categories)
    unknown_types = sorted(
        set(frame.loc[~frame["production_type"].isin(factor_categories), "production_type"])
    )
    if unknown_types:
        raise ValueError("unmapped ENTSO-E production types: " + ", ".join(unknown_types))
    frame["named_factor"] = frame["factor_category"].map(factors)
    missing_categories = sorted(
        set(
            frame.loc[
                frame["factor_category"].notna() & frame["named_factor"].isna(),
                "factor_category",
            ]
        )
    )
    if missing_categories:
        raise ValueError(
            "missing direct emission factor categories: "
            + ", ".join(missing_categories)
        )
    frame["uses_fallback"] = frame["factor_category"].isna()
    point_factor = frame["named_factor"].fillna(float(fallback["point_kg_co2e_per_mwh"]))
    lower_factor = frame["named_factor"].fillna(float(fallback["lower_kg_co2e_per_mwh"]))
    upper_factor = frame["named_factor"].fillna(float(fallback["upper_kg_co2e_per_mwh"]))
    frame["point_emissions"] = frame["generation_mwh"] * point_factor
    frame["lower_emissions"] = frame["generation_mwh"] * lower_factor
    frame["upper_emissions"] = frame["generation_mwh"] * upper_factor
    frame["named_generation"] = frame["generation_mwh"].where(~frame["uses_fallback"], 0.0)
    frame["fallback_generation"] = frame["generation_mwh"].where(frame["uses_fallback"], 0.0)

    hourly = frame.groupby(["timestamp_utc", "bidding_zone"], as_index=False).agg(
        total_generation_mwh=("generation_mwh", "sum"),
        named_factor_generation_mwh=("named_generation", "sum"),
        fallback_generation_mwh=("fallback_generation", "sum"),
        direct_emissions_kgco2e=("point_emissions", "sum"),
        direct_emissions_lower_kgco2e=("lower_emissions", "sum"),
        direct_emissions_upper_kgco2e=("upper_emissions", "sum"),
        production_type_count=("production_type", "nunique"),
        source_interval_count=("source_interval_count", "sum"),
    )
    denominator = hourly["total_generation_mwh"].replace(0, np.nan)
    hourly["named_factor_coverage_share"] = (
        hourly["named_factor_generation_mwh"] / denominator
    )
    hourly["direct_carbon_intensity_gco2_kwh"] = (
        hourly["direct_emissions_kgco2e"] / denominator
    )
    hourly["vintage_quality"] = vintage_quality
    hourly["compacted_at_utc"] = compacted_at or datetime.now(UTC)
    return hourly[
        [
            "timestamp_utc",
            "bidding_zone",
            "vintage_quality",
            "total_generation_mwh",
            "named_factor_generation_mwh",
            "fallback_generation_mwh",
            "named_factor_coverage_share",
            "direct_emissions_kgco2e",
            "direct_emissions_lower_kgco2e",
            "direct_emissions_upper_kgco2e",
            "direct_carbon_intensity_gco2_kwh",
            "production_type_count",
            "source_interval_count",
            "compacted_at_utc",
        ]
    ].sort_values(["timestamp_utc", "bidding_zone"]).reset_index(drop=True)


def build_neighbor_emissions_readiness(
    hourly: pd.DataFrame,
    contract: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Require every agreed zone and quantify factor-imputation exposure."""
    expected_hours = int((end - start).total_seconds() // 3600)
    coverage: dict[str, Any] = {}
    critical: list[str] = []
    for area in contract["areas"]:
        area_rows = hourly[hourly["bidding_zone"] == area]
        hourly_ratio = (
            area_rows["timestamp_utc"].nunique() / expected_hours
            if expected_hours
            else 0.0
        )
        named_ratio = (
            float(area_rows["named_factor_generation_mwh"].sum())
            / float(area_rows["total_generation_mwh"].sum())
            if float(area_rows["total_generation_mwh"].sum()) > 0
            else 0.0
        )
        coverage[area] = {
            "hourly_coverage": round(hourly_ratio, 4),
            "named_factor_generation_share": round(named_ratio, 4),
            "rows": int(len(area_rows)),
        }
        if hourly_ratio < float(contract["minimum_hourly_coverage"]):
            critical.append(f"low_hourly_coverage:{area}:{hourly_ratio:.4f}")
        if named_ratio < float(contract["minimum_named_factor_coverage"]):
            critical.append(f"high_fallback_factor_share:{area}:{1 - named_ratio:.4f}")
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contract_version": contract["contract_version"],
        "status": "fail" if critical else "pass",
        "interconnected_boundary_ready": not critical,
        "window": {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "expected_hours_per_area": expected_hours,
        },
        "critical_issues": critical,
        "coverage": coverage,
        "source_by_area": contract["area_sources"],
        "fallback_factor": contract["fallback_factor"],
        "known_limitations": contract.get("known_limitations", []),
    }


def validate_neighbor_emissions_schema(engine: Engine) -> None:
    tables = set(inspect(engine).get_table_names(schema="public"))
    if COMPACT_TABLE not in tables:
        raise RuntimeError(
            "Neighbor-emissions migration is incomplete. Apply db/causal_grid_compact.sql."
        )


def ingest_neighbor_emissions(
    api_token: str,
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    historical_backfill: bool = False,
    contract_path: str | Path = DEFAULT_CONFIG_PATH,
    archive_root: str | Path | None = None,
    storage: SupabaseStorage | None = None,
    client: EntsoePandasClient | None = None,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Fetch, validate, optionally archive, and upsert one bounded window."""
    validate_neighbor_emissions_schema(engine)
    contract = load_neighbor_emissions_contract(contract_path)
    start = utc_boundary(start)
    end = utc_boundary(end)
    generation = fetch_neighbor_generation(
        api_token,
        start,
        end,
        areas=tuple(contract["areas"]),
        client=client,
        area_sources=contract["area_sources"],
        elexon_fuel_type_mapping=contract["elexon_fuel_type_to_production_type"],
    )
    vintage = VINTAGE_HISTORICAL if historical_backfill else VINTAGE_OPERATIONAL
    hourly = build_neighbor_hourly_emissions(
        generation,
        contract,
        vintage_quality=vintage,
    )
    readiness = build_neighbor_emissions_readiness(hourly, contract, start, end)
    if not readiness["interconnected_boundary_ready"]:
        write_json(readiness, report_path)
        raise RuntimeError(
            "neighbor emissions quality failed: " + ", ".join(readiness["critical_issues"])
        )

    archive = None
    if archive_root is not None:
        archive = archive_generation_window(
            generation,
            start,
            end,
            vintage,
            Path(archive_root),
            storage,
        )
    rows = hourly.astype(object).where(pd.notna(hourly), None).to_dict(orient="records")
    upserted = upsert_neighbor_hourly_emissions(engine, rows)
    report = readiness | {
        "upserted_rows": upserted,
        "source_rows": int(len(generation)),
        "archive": archive,
    }
    write_json(report, report_path)
    return report


def archive_generation_window(
    generation: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    vintage: str,
    archive_root: Path,
    storage: SupabaseStorage | None,
) -> dict[str, Any]:
    """Archive normalized source rows before retaining only hourly compact outcomes."""
    month_key = start.strftime("%Y-%m")
    archive_dir = archive_root / vintage / month_key
    archive_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = archive_dir / "neighbor_generation_by_type.parquet"
    generation.to_parquet(parquet_path, index=False, compression="zstd")
    digest = sha256_file(parquet_path)
    manifest = {
        "format_version": "neighbor-emissions-archive-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "vintage_quality": vintage,
        "source_rows": int(len(generation)),
        "parquet_file": parquet_path.name,
        "parquet_bytes": parquet_path.stat().st_size,
        "sha256": digest,
    }
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    uploaded = False
    if storage is not None:
        storage.ensure_private_bucket()
        prefix = f"neighbor-emissions/{vintage}/{month_key}"
        storage.upload_verified(parquet_path, f"{prefix}/{parquet_path.name}", digest)
        storage.upload_verified(
            manifest_path,
            f"{prefix}/manifest.json",
            sha256_file(manifest_path),
        )
        uploaded = True
    return manifest | {
        "directory": str(archive_dir),
        "uploaded_to_supabase_storage": uploaded,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def utc_boundary(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ingest neighboring-zone actual generation and direct emissions."
    )
    parser.add_argument("--api-token", default=os.environ.get("ENTSOE_API_TOKEN"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--historical-backfill", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)
    start, end = build_window(args.start_date, args.end_date, args.lookback_days)
    if not args.end_date and not args.historical_backfill:
        end = min(end, pd.Timestamp.now(tz="UTC").floor("h"))
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                },
                indent=2,
            )
        )
        return 0
    if not args.api_token:
        parser.error("ENTSOE_API_TOKEN or --api-token is required")
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")
    report = ingest_neighbor_emissions(
        args.api_token,
        create_database_engine(args.database_url),
        start,
        end,
        historical_backfill=args.historical_backfill,
        report_path=args.report_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "upserted_rows": report["upserted_rows"],
                "report": args.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
