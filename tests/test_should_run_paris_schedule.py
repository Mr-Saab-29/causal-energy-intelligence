from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.should_run_paris_schedule import is_local_time_in_window


def test_manual_dispatch_always_runs() -> None:
    assert is_local_time_in_window(
        event_name="workflow_dispatch",
        window_start_hour=0,
        window_end_hour=6,
        timezone_name="Europe/Paris",
    )


def test_summer_utc_midnight_run_is_inside_paris_window() -> None:
    assert is_local_time_in_window(
        event_name="schedule",
        window_start_hour=0,
        window_end_hour=6,
        timezone_name="Europe/Paris",
        now_utc=datetime(2026, 8, 26, 0, 17, tzinfo=timezone.utc),
    )


def test_winter_utc_midnight_run_is_inside_paris_window() -> None:
    assert is_local_time_in_window(
        event_name="schedule",
        window_start_hour=0,
        window_end_hour=6,
        timezone_name="Europe/Paris",
        now_utc=datetime(2026, 1, 26, 0, 17, tzinfo=timezone.utc),
    )


def test_delayed_start_inside_paris_window_still_runs() -> None:
    assert is_local_time_in_window(
        event_name="schedule",
        window_start_hour=0,
        window_end_hour=6,
        timezone_name="Europe/Paris",
        now_utc=datetime(2026, 8, 26, 2, 30, tzinfo=timezone.utc),
    )


def test_start_after_paris_window_skips() -> None:
    assert not is_local_time_in_window(
        event_name="schedule",
        window_start_hour=0,
        window_end_hour=6,
        timezone_name="Europe/Paris",
        now_utc=datetime(2026, 8, 26, 4, 0, tzinfo=timezone.utc),
    )


def test_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        is_local_time_in_window(
            event_name="schedule",
            window_start_hour=6,
            window_end_hour=0,
            timezone_name="Europe/Paris",
            now_utc=datetime(2026, 8, 26, 0, 17, tzinfo=timezone.utc),
        )
