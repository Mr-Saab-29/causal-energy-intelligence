from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.causal_historical_backfill as backfill


class FakeStorage:
    def __init__(self) -> None:
        self.bucket_checked = False

    def ensure_private_bucket(self) -> None:
        self.bucket_checked = True


def test_iter_months_reverse_includes_both_boundaries() -> None:
    months = list(backfill.iter_months_reverse("2026-06-01", "2026-08-01"))

    assert [start.strftime("%Y-%m") for start, _ in months] == [
        "2026-08",
        "2026-07",
        "2026-06",
    ]
    assert months[-1][1] == pd.Timestamp("2026-07-01", tz="UTC")


def test_iter_months_reverse_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="through_month"):
        list(backfill.iter_months_reverse("2026-08-01", "2026-06-01"))


def test_runner_skips_complete_month_and_processes_next(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = FakeStorage()
    events: list[str] = []
    inspections = {"2026-07": 0}

    monkeypatch.setattr(backfill, "create_database_engine", lambda _: object())
    monkeypatch.setattr(backfill, "validate_causal_grid_schema", lambda _: None)
    monkeypatch.setattr(backfill, "apply_compact_schema", lambda _: None)

    def inspect_state(engine, active_storage, start, end):
        month = start.strftime("%Y-%m")
        if month == "2026-08":
            return {"month": month, "complete": True}
        inspections[month] += 1
        return {"month": month, "complete": inspections[month] > 1}

    monkeypatch.setattr(backfill, "inspect_month_state", inspect_state)

    def ingest(api_token, database_url, start, end, *, historical_backfill):
        events.append(f"ingest:{start.strftime('%Y-%m')}")
        return backfill.EntsoeCausalIngestionSummary(
            status="ok",
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
            historical_backfill=historical_backfill,
            france_area="FR",
            neighboring_areas=(),
        )

    monkeypatch.setattr(backfill, "ingest_entsoe_causal_data", ingest)

    def archive(engine, start, end, **kwargs):
        events.append(f"archive:{start.strftime('%Y-%m')}")
        assert kwargs["purge_raw"] is True
        return {
            "archive_bytes": 10,
            "compact_rows": {},
            "raw_rows_purged": {"grid_forecasts": 1},
        }

    monkeypatch.setattr(backfill, "archive_and_compact_month", archive)
    report_path = tmp_path / "report.json"

    report = backfill.run_historical_backfill(
        api_token="token",
        database_url="database",
        storage=storage,  # type: ignore[arg-type]
        from_month="2026-07-01",
        through_month="2026-08-01",
        report_path=report_path,
    )

    assert storage.bucket_checked is True
    assert events == ["ingest:2026-07", "archive:2026-07"]
    assert [month["status"] for month in report["months"]] == [
        "skipped_complete",
        "completed",
    ]
    assert json.loads(report_path.read_text())["status"] == "ok"


def test_sanitize_error_removes_entsoe_token() -> None:
    token = "private-token"
    error = RuntimeError(
        "request failed: https://example.test/api?securityToken=private-token&periodStart=1"
    )

    message = backfill.sanitize_error(error, token)

    assert token not in message
    assert "securityToken=[REDACTED]" in message
