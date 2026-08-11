from __future__ import annotations

from datetime import date

from src.data.local_ingest import resolve_refresh_start_date


def test_resolve_refresh_start_date_caps_empty_cache_with_lookback(tmp_path) -> None:
    result = resolve_refresh_start_date(
        target_end_date=date(2026, 8, 10),
        explicit_start_date=None,
        lookback_days=45,
        source_paths=[tmp_path / "missing.csv"],
    )

    assert result == date(2026, 6, 26)
