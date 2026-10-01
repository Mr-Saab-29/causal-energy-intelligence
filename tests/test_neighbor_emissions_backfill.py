from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.data.neighbor_emissions_backfill as backfill


class FakeStorage:
    def __init__(self) -> None:
        self.bucket_checked = False

    def ensure_private_bucket(self) -> None:
        self.bucket_checked = True


def test_runner_skips_durable_month_and_processes_incomplete_month(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = FakeStorage()
    inspections = {"2026-07": 0}
    events: list[str] = []
    monkeypatch.setattr(backfill, "create_database_engine", lambda _: object())
    monkeypatch.setattr(backfill, "validate_neighbor_emissions_schema", lambda _: None)

    def inspect_month(engine, active_storage, start, end, contract):
        month = start.strftime("%Y-%m")
        if month == "2026-08":
            return {"month": month, "complete": True}
        inspections[month] += 1
        return {"month": month, "complete": inspections[month] > 1}

    monkeypatch.setattr(backfill, "inspect_neighbor_month", inspect_month)

    def ingest(api_token, engine, start, end, **kwargs):
        events.append(start.strftime("%Y-%m"))
        return {
            "source_rows": 10,
            "upserted_rows": 6,
            "coverage": {},
            "archive": {},
        }

    monkeypatch.setattr(backfill, "ingest_neighbor_emissions", ingest)
    report_path = tmp_path / "report.json"
    report = backfill.run_neighbor_emissions_backfill(
        api_token="token",
        database_url="database",
        storage=storage,  # type: ignore[arg-type]
        from_month="2026-07-01",
        through_month="2026-08-01",
        archive_root=tmp_path / "archive",
        report_path=report_path,
    )

    assert storage.bucket_checked is True
    assert events == ["2026-07"]
    assert [item["status"] for item in report["months"]] == [
        "skipped_complete",
        "completed",
    ]
    assert json.loads(report_path.read_text())["status"] == "ok"
