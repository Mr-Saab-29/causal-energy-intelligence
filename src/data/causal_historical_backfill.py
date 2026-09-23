"""Resumable month-by-month ENTSO-E causal historical backfill."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.causal_archive_compact import (
    ARCHIVE_ROOT,
    DEFAULT_BUCKET,
    SupabaseStorage,
    apply_compact_schema,
    archive_and_compact_month,
)
from src.data.entsoe_causal_ingest import (
    EntsoeCausalIngestionSummary,
    REQUIRED_CAUSAL_GRID_TABLES,
    ingest_entsoe_causal_data,
    validate_causal_grid_schema,
)
from src.data.load import create_database_engine
from src.data.source_config import FRANCE_DIRECT_ENTSOE_NEIGHBORS, FRANCE_START_DATE

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_historical_backfill.json"


def iter_months_reverse(
    from_month: str | pd.Timestamp,
    through_month: str | pd.Timestamp,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield complete UTC calendar months from newest to oldest."""
    first = normalize_month(from_month)
    current = normalize_month(through_month)
    if current < first:
        raise ValueError("through_month must be on or after from_month")
    while current >= first:
        yield current, current + pd.offsets.MonthBegin(1)
        current -= pd.offsets.MonthBegin(1)


def normalize_month(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if timestamp.day != 1:
        raise ValueError("month boundaries must use the first day of a month")
    return timestamp.normalize()


def previous_complete_month() -> pd.Timestamp:
    current_month = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    return current_month - pd.offsets.MonthBegin(1)


def inspect_month_state(
    engine: Engine,
    storage: SupabaseStorage,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Check durable completion markers without trusting a local checkpoint."""
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": "historical_final",
    }
    raw_rows: dict[str, int] = {}
    with engine.connect() as connection:
        for table in REQUIRED_CAUSAL_GRID_TABLES:
            raw_rows[table] = int(
                connection.execute(
                    text(
                        f"""
                        select count(*) from {table}
                        where timestamp_utc >= :start and timestamp_utc < :end
                          and vintage_quality = :vintage
                        """
                    ),
                    params,
                ).scalar_one()
            )
        cross_rows = int(
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
        france_rows = int(
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
    compact_complete = (
        france_rows == expected_hours
        and cross_rows >= expected_hours * len(FRANCE_DIRECT_ENTSOE_NEIGHBORS)
    )
    month_key = start.strftime("%Y-%m")
    manifest_exists = storage.object_exists(
        f"historical_final/{month_key}/manifest.json"
    )
    raw_rows_remaining = sum(raw_rows.values())
    return {
        "month": month_key,
        "complete": manifest_exists and compact_complete and raw_rows_remaining == 0,
        "manifest_exists": manifest_exists,
        "compact_complete": compact_complete,
        "raw_rows_remaining": raw_rows_remaining,
        "raw_rows": raw_rows,
        "compact_rows": {
            "causal_cross_border_hourly_features": cross_rows,
            "causal_france_hourly_features": france_rows,
        },
    }


def run_historical_backfill(
    *,
    api_token: str,
    database_url: str,
    storage: SupabaseStorage,
    from_month: str | pd.Timestamp,
    through_month: str | pd.Timestamp,
    archive_root: str | Path = ARCHIVE_ROOT,
    purge_raw: bool = True,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Process and checkpoint every requested historical month."""
    engine = create_database_engine(database_url)
    validate_causal_grid_schema(engine)
    apply_compact_schema(engine)
    storage.ensure_private_bucket()
    months = list(iter_months_reverse(from_month, through_month))
    report: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "from_month": months[-1][0].strftime("%Y-%m"),
        "through_month": months[0][0].strftime("%Y-%m"),
        "purge_raw": purge_raw,
        "months": [],
    }
    persist_report(report, report_path)

    try:
        for start, end in months:
            state = inspect_month_state(engine, storage, start, end)
            if state["complete"]:
                result = {"month": state["month"], "status": "skipped_complete"}
            else:
                ingestion = ingest_entsoe_causal_data(
                    api_token,
                    database_url,
                    start,
                    end,
                    historical_backfill=True,
                )
                archive = archive_and_compact_month(
                    engine,
                    start,
                    end,
                    vintage_quality="historical_final",
                    archive_root=archive_root,
                    storage=storage,
                    purge_raw=purge_raw,
                )
                final_state = inspect_month_state(engine, storage, start, end)
                if purge_raw and not final_state["complete"]:
                    raise RuntimeError(
                        f"month {state['month']} did not reach a durable completed state"
                    )
                result = {
                    "month": state["month"],
                    "status": "completed",
                    "ingestion": asdict(ingestion),
                    "archive_bytes": archive["archive_bytes"],
                    "compact_rows": archive["compact_rows"],
                    "raw_rows_purged": archive["raw_rows_purged"],
                }
            report["months"].append(result)
            persist_report(report, report_path)
            print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        safe_error = sanitize_error(error, api_token)
        report["status"] = "failed"
        report["failed_at_utc"] = datetime.now(UTC).isoformat()
        report["error"] = safe_error
        persist_report(report, report_path)
        raise RuntimeError(safe_error) from None

    report["status"] = "ok"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    persist_report(report, report_path)
    return report


def persist_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)


def sanitize_error(error: Exception, api_token: str) -> str:
    message = f"{type(error).__name__}: {error}"
    if api_token:
        message = message.replace(api_token, "[REDACTED]")
    return re.sub(
        r"([?&]securityToken=)[^&\s]+",
        r"\1[REDACTED]",
        message,
        flags=re.IGNORECASE,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run the resumable monthly ENTSO-E causal historical backfill."
    )
    parser.add_argument("--api-token", default=os.environ.get("ENTSOE_API_TOKEN"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--project-url", default=os.environ.get("SUPABASE_PROJECT_URL")
    )
    parser.add_argument(
        "--service-key", default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )
    parser.add_argument(
        "--bucket", default=os.environ.get("CAUSAL_ARCHIVE_BUCKET", DEFAULT_BUCKET)
    )
    parser.add_argument("--from-month", default=FRANCE_START_DATE.strftime("%Y-%m"))
    parser.add_argument(
        "--through-month", default=previous_complete_month().strftime("%Y-%m")
    )
    parser.add_argument("--archive-root", default=str(ARCHIVE_ROOT))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-purge", action="store_true")
    args = parser.parse_args(argv)

    required = {
        "ENTSOE_API_TOKEN": args.api_token,
        "DATABASE_URL": args.database_url,
        "SUPABASE_PROJECT_URL": args.project_url,
        "SUPABASE_SERVICE_ROLE_KEY": args.service_key,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing required configuration: " + ", ".join(missing))

    storage = SupabaseStorage(args.project_url, args.service_key, args.bucket)
    try:
        report = run_historical_backfill(
            api_token=args.api_token,
            database_url=args.database_url,
            storage=storage,
            from_month=f"{args.from_month}-01",
            through_month=f"{args.through_month}-01",
            archive_root=args.archive_root,
            purge_raw=not args.no_purge,
            report_path=args.report_path,
        )
    finally:
        storage.close()
    print(
        json.dumps(
            {
                "status": report["status"],
                "months": len(report["months"]),
                "report": args.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
