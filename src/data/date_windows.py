"""Date-window helpers for bounded historical API ingestion."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class DateWindow:
    """Inclusive date-window used for API extraction."""

    start_date: date
    end_date: date


def iter_date_windows(
    start_date: date,
    end_date: date,
    window_days: int,
) -> Iterator[DateWindow]:
    """Yield inclusive date windows from start_date through end_date."""
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date")

    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=window_days - 1), end_date)
        yield DateWindow(start_date=current_start, end_date=current_end)
        current_start = current_end + timedelta(days=1)


def iter_calendar_year_windows(
    start_date: date,
    end_date: date,
) -> Iterator[DateWindow]:
    """Yield inclusive windows split on calendar-year boundaries."""
    if end_date < start_date:
        raise ValueError("end_date must be greater than or equal to start_date")

    current_start = start_date
    while current_start <= end_date:
        current_end = min(date(current_start.year, 12, 31), end_date)
        yield DateWindow(start_date=current_start, end_date=current_end)
        current_start = current_end + timedelta(days=1)
