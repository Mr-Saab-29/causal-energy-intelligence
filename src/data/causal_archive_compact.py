"""Archive native causal data and compact it into hourly Postgres features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from src.data.causal_data_quality import (
    build_causal_data_quality_report,
    default_quality_window,
    normalize_boundary,
)
from src.data.entsoe_causal_ingest import (
    REQUIRED_CAUSAL_GRID_TABLES,
    validate_causal_grid_schema,
)
from src.data.load import create_database_engine
from src.data.source_config import FRANCE_DIRECT_ENTSOE_NEIGHBORS

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "data/archive/entsoe"
COMPACT_SCHEMA_PATH = ROOT / "db/causal_grid_compact.sql"
DEFAULT_BUCKET = "causal-grid-archive"
COMPACT_TABLES = (
    "causal_cross_border_hourly_features",
    "causal_france_hourly_features",
)


def archive_and_compact_month(
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    vintage_quality: str = "historical_final",
    archive_root: str | Path = ARCHIVE_ROOT,
    storage: SupabaseStorage | None = None,
    purge_raw: bool = False,
    apply_schema: bool = False,
) -> dict[str, Any]:
    """Validate, archive, compact, optionally upload, then safely purge one month."""
    start = normalize_boundary(start)
    end = normalize_boundary(end)
    if end <= start:
        raise ValueError("end must be after start")
    if purge_raw and storage is None:
        raise ValueError("purge_raw requires verified Supabase Storage upload")
    if purge_raw and vintage_quality != "historical_final":
        raise ValueError("only historical_final staging rows may be purged")

    validate_causal_grid_schema(engine)
    if apply_schema:
        apply_compact_schema(engine)
    validate_compact_schema(engine)

    month_key = start.strftime("%Y-%m")
    quality_path = ROOT / f"reports/metrics/causal_data_quality_{month_key}.json"
    quality = build_causal_data_quality_report(
        engine,
        start,
        end,
        vintage_quality=vintage_quality,
        max_projected_storage_bytes=10 * 1024**3,
        output_path=quality_path,
    )
    if not quality["backfill_ready"]:
        raise RuntimeError(
            "causal-data quality failed before archival: "
            + ", ".join(quality["critical_issues"])
        )

    archive_dir = Path(archive_root) / vintage_quality / month_key
    archive_dir.mkdir(parents=True, exist_ok=True)
    files = export_month_to_parquet(
        engine,
        start,
        end,
        vintage_quality,
        archive_dir,
    )
    compact_counts = compact_month(engine, start, end, vintage_quality)
    manifest = build_manifest(
        start,
        end,
        vintage_quality,
        files,
        compact_counts,
    )
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    uploaded = False
    if storage is not None:
        storage.ensure_private_bucket()
        object_prefix = f"{vintage_quality}/{month_key}"
        for item in files:
            storage.upload_verified(
                item["path"],
                f"{object_prefix}/{Path(item['path']).name}",
                item["sha256"],
            )
        storage.upload_verified(
            manifest_path,
            f"{object_prefix}/manifest.json",
            sha256_file(manifest_path),
        )
        uploaded = True

    purged_rows: dict[str, int] = {}
    if purge_raw:
        purged_rows = purge_archived_month(
            engine,
            start,
            end,
            vintage_quality,
        )

    return {
        "status": "ok",
        "window": {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "vintage_quality": vintage_quality,
        },
        "quality_status": quality["status"],
        "archive_directory": str(archive_dir),
        "archive_files": files,
        "archive_bytes": sum(int(item["size_bytes"]) for item in files),
        "compact_rows": compact_counts,
        "uploaded_to_supabase_storage": uploaded,
        "raw_rows_purged": purged_rows,
        "manifest": str(manifest_path),
    }


def export_month_to_parquet(
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    vintage_quality: str,
    archive_dir: Path,
) -> list[dict[str, Any]]:
    """Write deterministic monthly table extracts as compressed Parquet."""
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": vintage_quality,
    }
    files: list[dict[str, Any]] = []
    with engine.connect() as connection:
        for table in REQUIRED_CAUSAL_GRID_TABLES:
            frame = pd.read_sql_query(
                text(
                    f"""
                    select * from {table}
                    where timestamp_utc >= :start and timestamp_utc < :end
                      and vintage_quality = :vintage
                    order by timestamp_utc, source_record_id
                    """
                ),
                connection,
                params=params,
            )
            frame = normalize_parquet_values(frame)
            path = archive_dir / f"{table}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            files.append(
                {
                    "table": table,
                    "path": str(path),
                    "row_count": int(len(frame)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def normalize_parquet_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert database UUID objects to portable Parquet string values."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        values = result[column].dropna()
        if not values.empty and values.map(lambda value: isinstance(value, UUID)).any():
            result[column] = result[column].map(
                lambda value: str(value) if isinstance(value, UUID) else value
            )
    return result


def compact_month(
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    vintage_quality: str,
) -> dict[str, int]:
    """Upsert hourly neighbor and France-level features for one month."""
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": vintage_quality,
    }
    with engine.begin() as connection:
        connection.execute(text(CROSS_BORDER_COMPACTION_SQL), params)
        connection.execute(text(FRANCE_COMPACTION_SQL), params)
        cross_count = int(
            connection.execute(
                text(
                    """
                    select count(*) from causal_cross_border_hourly_features
                    where timestamp_utc >= :start and timestamp_utc < :end
                      and vintage_quality = :vintage
                    """
                ),
                params,
            ).scalar_one()
        )
        france_count = int(
            connection.execute(
                text(
                    """
                    select count(*) from causal_france_hourly_features
                    where timestamp_utc >= :start and timestamp_utc < :end
                      and vintage_quality = :vintage
                    """
                ),
                params,
            ).scalar_one()
        )
    expected_hours = int((end - start).total_seconds() // 3600)
    if france_count != expected_hours:
        raise RuntimeError(
            f"compact France rows incomplete: {france_count} != {expected_hours}"
        )
    expected_cross_rows = expected_hours * len(FRANCE_DIRECT_ENTSOE_NEIGHBORS)
    if cross_count < expected_cross_rows:
        raise RuntimeError(
            f"compact cross-border rows incomplete: {cross_count} < {expected_cross_rows}"
        )
    return {
        "causal_cross_border_hourly_features": cross_count,
        "causal_france_hourly_features": france_count,
    }


def purge_archived_month(
    engine: Engine,
    start: pd.Timestamp,
    end: pd.Timestamp,
    vintage_quality: str,
) -> dict[str, int]:
    """Delete only the verified monthly staging rows and make pages reusable."""
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": vintage_quality,
    }
    deleted: dict[str, int] = {}
    with engine.begin() as connection:
        for table in REQUIRED_CAUSAL_GRID_TABLES:
            result = connection.execute(
                text(
                    f"""
                    delete from {table}
                    where timestamp_utc >= :start and timestamp_utc < :end
                      and vintage_quality = :vintage
                    """
                ),
                params,
            )
            deleted[table] = int(result.rowcount or 0)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        for table in REQUIRED_CAUSAL_GRID_TABLES:
            connection.execute(text(f"vacuum (analyze) {table}"))
    return deleted


def apply_compact_schema(engine: Engine) -> None:
    """Apply the idempotent compact-table migration."""
    sql = COMPACT_SCHEMA_PATH.read_text(encoding="utf-8")
    raw_connection = engine.raw_connection()
    try:
        cursor = raw_connection.cursor()
        cursor.execute(sql)
        raw_connection.commit()
        cursor.close()
    finally:
        raw_connection.close()


def validate_compact_schema(engine: Engine) -> None:
    existing = set(inspect(engine).get_table_names(schema="public"))
    missing = [table for table in COMPACT_TABLES if table not in existing]
    if missing:
        raise RuntimeError(
            "Compact causal migration is incomplete. Missing tables: "
            f"{', '.join(missing)}. Apply db/causal_grid_compact.sql."
        )


def build_manifest(
    start: pd.Timestamp,
    end: pd.Timestamp,
    vintage_quality: str,
    files: list[dict[str, Any]],
    compact_counts: dict[str, int],
) -> dict[str, Any]:
    portable_files = [
        {key: value for key, value in item.items() if key != "path"}
        | {"file": Path(item["path"]).name}
        for item in files
    ]
    return {
        "format_version": "causal-grid-archive-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "vintage_quality": vintage_quality,
        "files": portable_files,
        "compact_rows": compact_counts,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SupabaseStorage:
    """Minimal private-bucket client that never logs credentials."""

    def __init__(
        self,
        project_url: str,
        service_role_key: str,
        bucket: str = DEFAULT_BUCKET,
    ) -> None:
        if not project_url or not service_role_key:
            raise ValueError("Supabase project URL and service role key are required")
        self.bucket = bucket
        headers = {"apikey": service_role_key}
        if not service_role_key.startswith("sb_secret_"):
            headers["authorization"] = f"Bearer {service_role_key}"
        self.client = httpx.Client(
            base_url=project_url.rstrip("/"),
            headers=headers,
            timeout=120.0,
        )

    def ensure_private_bucket(self) -> None:
        response = self.client.get(f"/storage/v1/bucket/{quote(self.bucket)}")
        if self._bucket_not_found(response):
            response = self.client.post(
                "/storage/v1/bucket",
                json={
                    "id": self.bucket,
                    "name": self.bucket,
                    "public": False,
                    "allowed_mime_types": [
                        "application/vnd.apache.parquet",
                        "application/json",
                    ],
                },
            )
        self._raise_for_status(response, "bucket setup")

    @staticmethod
    def _bucket_not_found(response: httpx.Response) -> bool:
        if response.status_code == 404:
            return True
        try:
            body = response.json()
        except ValueError:
            return False
        return body.get("code") == "NoSuchBucket" or str(body.get("statusCode")) == "404"

    def upload_verified(
        self,
        local_path: str | Path,
        object_path: str,
        expected_sha256: str,
    ) -> None:
        path = Path(local_path)
        content_type = (
            "application/json"
            if path.suffix == ".json"
            else "application/vnd.apache.parquet"
        )
        encoded_path = quote(object_path, safe="/")
        response = self.client.post(
            f"/storage/v1/object/{quote(self.bucket)}/{encoded_path}",
            content=path.read_bytes(),
            headers={"content-type": content_type, "x-upsert": "true"},
        )
        self._raise_for_status(response, f"upload {object_path}")
        downloaded = self.client.get(
            f"/storage/v1/object/authenticated/{quote(self.bucket)}/{encoded_path}"
        )
        self._raise_for_status(downloaded, f"verify {object_path}")
        actual_sha256 = hashlib.sha256(downloaded.content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"storage checksum mismatch for {object_path}")

    def object_exists(self, object_path: str) -> bool:
        encoded_path = quote(object_path, safe="/")
        response = self.client.get(
            f"/storage/v1/object/authenticated/{quote(self.bucket)}/{encoded_path}"
        )
        if response.is_success:
            return True
        if self._object_not_found(response):
            return False
        self._raise_for_status(response, f"inspect {object_path}")
        return False

    @staticmethod
    def _object_not_found(response: httpx.Response) -> bool:
        if response.status_code == 404:
            return True
        try:
            body = response.json()
        except ValueError:
            return False
        return body.get("code") in {"NoSuchKey", "NoSuchObject"} or str(
            body.get("statusCode")
        ) == "404"

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if response.is_success:
            return
        raise RuntimeError(
            f"Supabase Storage {operation} failed with HTTP {response.status_code}"
        )


CROSS_BORDER_COMPACTION_SQL = """
with directional_hourly as (
    select
        date_trunc('hour', timestamp_utc) as timestamp_utc,
        case
            when from_bidding_zone = 'FR' then to_bidding_zone
            else from_bidding_zone
        end as neighbor_bidding_zone,
        metric,
        case when to_bidding_zone = 'FR' then 'import' else 'export' end as direction,
        avg(value_mw)::double precision as value_mw,
        count(*) as interval_count
    from cross_border_observations
    where timestamp_utc >= :start and timestamp_utc < :end
      and vintage_quality = :vintage
      and (from_bidding_zone = 'FR' or to_bidding_zone = 'FR')
    group by 1, 2, 3, 4
), pivoted as (
    select
        timestamp_utc,
        neighbor_bidding_zone,
        max(value_mw) filter (where metric = 'physical_flow' and direction = 'import')
            as physical_import_mw,
        max(value_mw) filter (where metric = 'physical_flow' and direction = 'export')
            as physical_export_mw,
        max(value_mw) filter (where metric = 'scheduled_exchange' and direction = 'import')
            as scheduled_import_mw,
        max(value_mw) filter (where metric = 'scheduled_exchange' and direction = 'export')
            as scheduled_export_mw,
        max(value_mw) filter (where metric = 'day_ahead_capacity' and direction = 'import')
            as day_ahead_import_capacity_mw,
        max(value_mw) filter (where metric = 'day_ahead_capacity' and direction = 'export')
            as day_ahead_export_capacity_mw,
        sum(interval_count)::integer as source_interval_count
    from directional_hourly
    group by timestamp_utc, neighbor_bidding_zone
)
insert into causal_cross_border_hourly_features (
    timestamp_utc, neighbor_bidding_zone, vintage_quality,
    physical_import_mw, physical_export_mw, net_physical_import_mw,
    scheduled_import_mw, scheduled_export_mw, net_scheduled_import_mw,
    day_ahead_import_capacity_mw, day_ahead_export_capacity_mw,
    source_interval_count, compacted_at_utc
)
select
    timestamp_utc, neighbor_bidding_zone, :vintage,
    physical_import_mw, physical_export_mw,
    coalesce(physical_import_mw, 0) - coalesce(physical_export_mw, 0),
    scheduled_import_mw, scheduled_export_mw,
    coalesce(scheduled_import_mw, 0) - coalesce(scheduled_export_mw, 0),
    day_ahead_import_capacity_mw, day_ahead_export_capacity_mw,
    source_interval_count, now()
from pivoted
on conflict (timestamp_utc, neighbor_bidding_zone, vintage_quality)
do update set
    physical_import_mw = excluded.physical_import_mw,
    physical_export_mw = excluded.physical_export_mw,
    net_physical_import_mw = excluded.net_physical_import_mw,
    scheduled_import_mw = excluded.scheduled_import_mw,
    scheduled_export_mw = excluded.scheduled_export_mw,
    net_scheduled_import_mw = excluded.net_scheduled_import_mw,
    day_ahead_import_capacity_mw = excluded.day_ahead_import_capacity_mw,
    day_ahead_export_capacity_mw = excluded.day_ahead_export_capacity_mw,
    source_interval_count = excluded.source_interval_count,
    compacted_at_utc = now()
"""


FRANCE_COMPACTION_SQL = """
with hours as (
    select generate_series(
        cast(:start as timestamptz),
        cast(:end as timestamptz) - interval '1 hour',
        interval '1 hour'
    ) as timestamp_utc
), forecast_hourly as (
    select
        date_trunc('hour', timestamp_utc) as timestamp_utc,
        avg(forecast_mw) filter (where forecast_type = 'load')::double precision
            as load_forecast_mw,
        avg(forecast_mw) filter (where forecast_type = 'wind_onshore')::double precision
            as wind_onshore_forecast_mw,
        avg(forecast_mw) filter (where forecast_type = 'wind_offshore')::double precision
            as wind_offshore_forecast_mw,
        avg(forecast_mw) filter (where forecast_type = 'solar')::double precision
            as solar_forecast_mw,
        count(*)::integer as forecast_interval_count
    from grid_forecasts
    where timestamp_utc >= :start and timestamp_utc < :end
      and vintage_quality = :vintage
    group by 1
), latest_outages as (
    select distinct on (outage_mrid)
        outage_mrid, outage_type, timestamp_utc, end_utc, unavailable_capacity_mw
    from generation_outages
    where timestamp_utc < :end and end_utc > :start
      and vintage_quality = :vintage
    order by outage_mrid, revision_number desc, publication_timestamp_utc desc nulls last
), outage_hourly as (
    select
        hours.timestamp_utc,
        sum(coalesce(outage.unavailable_capacity_mw, 0)) filter (
            where outage.outage_type = 'planned'
        )::double precision as planned_unavailable_capacity_mw,
        sum(coalesce(outage.unavailable_capacity_mw, 0)) filter (
            where outage.outage_type = 'unplanned'
        )::double precision as unplanned_unavailable_capacity_mw,
        count(outage.outage_mrid) filter (where outage.outage_type = 'planned')::integer
            as planned_outage_count,
        count(outage.outage_mrid) filter (where outage.outage_type = 'unplanned')::integer
            as unplanned_outage_count
    from hours
    left join latest_outages as outage
      on outage.timestamp_utc < hours.timestamp_utc + interval '1 hour'
     and outage.end_utc > hours.timestamp_utc
    group by hours.timestamp_utc
), balancing_hourly as (
    select
        date_trunc('hour', timestamp_utc) as timestamp_utc,
        avg(value) filter (
            where metric = 'imbalance_price' and direction = 'up'
        )::double precision as imbalance_price_up_eur_mwh,
        avg(value) filter (
            where metric = 'imbalance_price' and direction = 'down'
        )::double precision as imbalance_price_down_eur_mwh,
        avg(value) filter (where metric = 'imbalance_volume')::double precision
            as imbalance_volume_mwh,
        avg(value) filter (where metric = 'activated_energy_price')::double precision
            as activated_energy_price_eur_mwh,
        count(*)::integer as balancing_interval_count
    from balancing_observations
    where timestamp_utc >= :start and timestamp_utc < :end
      and vintage_quality = :vintage
    group by 1
)
insert into causal_france_hourly_features (
    timestamp_utc, vintage_quality,
    load_forecast_mw, wind_onshore_forecast_mw,
    wind_offshore_forecast_mw, solar_forecast_mw,
    planned_unavailable_capacity_mw, unplanned_unavailable_capacity_mw,
    planned_outage_count, unplanned_outage_count,
    imbalance_price_up_eur_mwh, imbalance_price_down_eur_mwh,
    imbalance_volume_mwh, activated_energy_price_eur_mwh,
    forecast_interval_count, balancing_interval_count, compacted_at_utc
)
select
    hours.timestamp_utc, :vintage,
    forecast.load_forecast_mw, forecast.wind_onshore_forecast_mw,
    forecast.wind_offshore_forecast_mw, forecast.solar_forecast_mw,
    coalesce(outage.planned_unavailable_capacity_mw, 0),
    coalesce(outage.unplanned_unavailable_capacity_mw, 0),
    coalesce(outage.planned_outage_count, 0),
    coalesce(outage.unplanned_outage_count, 0),
    balancing.imbalance_price_up_eur_mwh,
    balancing.imbalance_price_down_eur_mwh,
    balancing.imbalance_volume_mwh,
    balancing.activated_energy_price_eur_mwh,
    coalesce(forecast.forecast_interval_count, 0),
    coalesce(balancing.balancing_interval_count, 0),
    now()
from hours
left join forecast_hourly as forecast using (timestamp_utc)
left join outage_hourly as outage using (timestamp_utc)
left join balancing_hourly as balancing using (timestamp_utc)
on conflict (timestamp_utc, vintage_quality)
do update set
    load_forecast_mw = excluded.load_forecast_mw,
    wind_onshore_forecast_mw = excluded.wind_onshore_forecast_mw,
    wind_offshore_forecast_mw = excluded.wind_offshore_forecast_mw,
    solar_forecast_mw = excluded.solar_forecast_mw,
    planned_unavailable_capacity_mw = excluded.planned_unavailable_capacity_mw,
    unplanned_unavailable_capacity_mw = excluded.unplanned_unavailable_capacity_mw,
    planned_outage_count = excluded.planned_outage_count,
    unplanned_outage_count = excluded.unplanned_outage_count,
    imbalance_price_up_eur_mwh = excluded.imbalance_price_up_eur_mwh,
    imbalance_price_down_eur_mwh = excluded.imbalance_price_down_eur_mwh,
    imbalance_volume_mwh = excluded.imbalance_volume_mwh,
    activated_energy_price_eur_mwh = excluded.activated_energy_price_eur_mwh,
    forecast_interval_count = excluded.forecast_interval_count,
    balancing_interval_count = excluded.balancing_interval_count,
    compacted_at_utc = now()
"""


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    default_start, default_end = default_quality_window()
    parser = argparse.ArgumentParser(
        description="Archive and compact one causal-grid month."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--start-date", default=default_start.date().isoformat())
    parser.add_argument("--end-date", default=default_end.date().isoformat())
    parser.add_argument(
        "--vintage-quality",
        choices=("historical_final", "operational_snapshot"),
        default="historical_final",
    )
    parser.add_argument("--archive-root", default=str(ARCHIVE_ROOT))
    parser.add_argument(
        "--bucket", default=os.environ.get("CAUSAL_ARCHIVE_BUCKET", DEFAULT_BUCKET)
    )
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--purge-raw", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")

    storage = None
    if not args.skip_upload:
        project_url = os.environ.get("SUPABASE_PROJECT_URL")
        service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not project_url or not service_key:
            parser.error(
                "SUPABASE_PROJECT_URL and SUPABASE_SERVICE_ROLE_KEY are required "
                "unless --skip-upload is used"
            )
        storage = SupabaseStorage(project_url, service_key, args.bucket)

    try:
        summary = archive_and_compact_month(
            create_database_engine(args.database_url),
            normalize_boundary(args.start_date),
            normalize_boundary(args.end_date),
            vintage_quality=args.vintage_quality,
            archive_root=args.archive_root,
            storage=storage,
            purge_raw=args.purge_raw,
            apply_schema=args.apply_schema,
        )
    finally:
        if storage is not None:
            storage.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
