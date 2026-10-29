"""Focused checks for post-ingestion freshness diagnostics."""

from datetime import UTC, datetime

import pandas as pd

from src.data.supabase_ingest import evaluate_feature_freshness


def test_feature_freshness_reports_a_dst_gap_without_truncating_later_hours() -> None:
    timestamps = pd.date_range("2026-10-24T00:00:00Z", "2026-10-26T23:00:00Z", freq="h")
    timestamps = timestamps[timestamps != pd.Timestamp("2026-10-25T00:00:00Z")]

    result = evaluate_feature_freshness(
        pd.DataFrame({"timestamp_utc": timestamps}),
        as_of=datetime(2026, 10, 27, tzinfo=UTC),
    )

    assert result["latest_feature_timestamp_utc"] == "2026-10-26T23:00:00+00:00"
    assert result["feature_freshness_lag_hours"] == 1.0
    assert result["recent_missing_feature_hours"] == 1


def test_feature_freshness_exposes_a_stale_joined_endpoint() -> None:
    result = evaluate_feature_freshness(
        pd.DataFrame({"timestamp_utc": ["2026-10-24T23:00:00Z"]}),
        as_of=datetime(2026, 10, 28, 6, 35, tzinfo=UTC),
    )

    assert result["feature_freshness_lag_hours"] == 79.58
