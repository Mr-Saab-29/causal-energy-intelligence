"""Resumable monthly backfill for interconnected-zone emissions outcomes."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.causal_archive_compact import DEFAULT_BUCKET, SupabaseStorage
from src.data.causal_historical_backfill import (
    iter_months_reverse,
    persist_report,
    previous_complete_month,
    sanitize_error,
)
from src.data.load import create_database_engine
from src.data.neighbor_emissions import (
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_CONFIG_PATH,
    ingest_neighbor_emissions,
    load_neighbor_emissions_contract,
    validate_neighbor_emissions_schema,
)
from src.data.source_config import FRANCE_START_DATE

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/neighbor_emissions_backfill.json"


def inspect_neighbor_month(
    engine: Engine,
    storage: SupabaseStorage,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Inspect compact rows and the verified archive completion marker."""
    params = {
        "start": start.to_pydatetime(),
        "end": end.to_pydatetime(),
        "vintage": "historical_final",
    }
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                select bidding_zone, count(distinct timestamp_utc) as row_count
                from causal_neighbor_hourly_emissions
                where timestamp_utc >= :start and timestamp_utc < :end
                  and vintage_quality = :vintage
                group by bidding_zone
                """
            ),
            params,
        ).mappings().all()
    counts = {str(row["bidding_zone"]): int(row["row_count"]) for row in rows}
    expected_hours = int((end - start).total_seconds() // 3600)
    minimum_rows = math.ceil(
        expected_hours * float(contract["minimum_hourly_coverage"])
    )
    compact_complete = all(counts.get(area, 0) >= minimum_rows for area in contract["areas"])
    month_key = start.strftime("%Y-%m")
    manifest_path = f"neighbor-emissions/historical_final/{month_key}/manifest.json"
    manifest_exists = storage.object_exists(manifest_path)
    return {
        "month": month_key,
        "complete": compact_complete and manifest_exists,
        "compact_complete": compact_complete,
        "manifest_exists": manifest_exists,
        "expected_hours": expected_hours,
        "minimum_rows": minimum_rows,
        "rows_by_area": counts,
    }


def run_neighbor_emissions_backfill(
    *,
    api_token: str,
    database_url: str,
    storage: SupabaseStorage,
    from_month: str | pd.Timestamp,
    through_month: str | pd.Timestamp,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Backfill each month and resume only from durable compact/archive state."""
    engine = create_database_engine(database_url)
    validate_neighbor_emissions_schema(engine)
    storage.ensure_private_bucket()
    contract = load_neighbor_emissions_contract(DEFAULT_CONFIG_PATH)
    months = list(iter_months_reverse(from_month, through_month))
    report: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "contract_version": contract["contract_version"],
        "from_month": months[-1][0].strftime("%Y-%m"),
        "through_month": months[0][0].strftime("%Y-%m"),
        "months": [],
    }
    persist_report(report, report_path)
    try:
        for start, end in months:
            state = inspect_neighbor_month(engine, storage, start, end, contract)
            if state["complete"]:
                result = {"month": state["month"], "status": "skipped_complete"}
            else:
                month_report_path = (
                    ROOT
                    / f"reports/metrics/neighbor_emissions_readiness_{state['month']}.json"
                )
                ingestion = ingest_neighbor_emissions(
                    api_token,
                    engine,
                    start,
                    end,
                    historical_backfill=True,
                    archive_root=archive_root,
                    storage=storage,
                    report_path=month_report_path,
                )
                final_state = inspect_neighbor_month(engine, storage, start, end, contract)
                if not final_state["complete"]:
                    raise RuntimeError(
                        f"neighbor emissions month {state['month']} did not reach "
                        "durable completion"
                    )
                result = {
                    "month": state["month"],
                    "status": "completed",
                    "source_rows": ingestion["source_rows"],
                    "compact_rows": ingestion["upserted_rows"],
                    "coverage": ingestion["coverage"],
                    "archive": ingestion["archive"],
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


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Backfill neighboring-zone direct emissions month by month."
    )
    parser.add_argument("--api-token", default=os.environ.get("ENTSOE_API_TOKEN"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--project-url", default=os.environ.get("SUPABASE_PROJECT_URL"))
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
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
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
        report = run_neighbor_emissions_backfill(
            api_token=args.api_token,
            database_url=args.database_url,
            storage=storage,
            from_month=f"{args.from_month}-01",
            through_month=f"{args.through_month}-01",
            archive_root=args.archive_root,
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
