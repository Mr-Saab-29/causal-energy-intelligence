from __future__ import annotations

from datetime import date

import pytest

from scripts.should_run_paris_schedule import is_intended_local_hour, parse_cron_minute_hour


def test_manual_dispatch_always_runs() -> None:
    assert is_intended_local_hour(
        event_name="workflow_dispatch",
        schedule="",
        target_hour=2,
        timezone_name="Europe/Paris",
    )


def test_summer_uses_utc_midnight_schedule_for_paris_02() -> None:
    assert is_intended_local_hour(
        event_name="schedule",
        schedule="17 0 * * *",
        target_hour=2,
        timezone_name="Europe/Paris",
        utc_date=date(2026, 8, 26),
    )
    assert not is_intended_local_hour(
        event_name="schedule",
        schedule="17 1 * * *",
        target_hour=2,
        timezone_name="Europe/Paris",
        utc_date=date(2026, 8, 26),
    )


def test_winter_uses_utc_01_schedule_for_paris_02() -> None:
    assert not is_intended_local_hour(
        event_name="schedule",
        schedule="17 0 * * *",
        target_hour=2,
        timezone_name="Europe/Paris",
        utc_date=date(2026, 1, 26),
    )
    assert is_intended_local_hour(
        event_name="schedule",
        schedule="17 1 * * *",
        target_hour=2,
        timezone_name="Europe/Paris",
        utc_date=date(2026, 1, 26),
    )


def test_parse_cron_minute_hour_rejects_combined_hours() -> None:
    with pytest.raises(ValueError):
        parse_cron_minute_hour("17 0,1 * * *")
