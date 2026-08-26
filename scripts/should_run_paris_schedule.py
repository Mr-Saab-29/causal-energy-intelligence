"""Decide whether a delayed GitHub schedule is the intended Paris run."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> int:
    """Write a GitHub Actions should_run output for the daily schedule guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--schedule", default="")
    parser.add_argument("--target-hour", type=int, default=2)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--utc-date", default=None)
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args()

    should_run = is_intended_local_hour(
        event_name=args.event_name,
        schedule=args.schedule,
        target_hour=args.target_hour,
        timezone_name=args.timezone,
        utc_date=parse_utc_date(args.utc_date),
    )
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"should_run={str(should_run).lower()}\n")
    print(
        "Running scheduled workflow."
        if should_run
        else f"Skipping because the scheduled UTC time is not {args.target_hour:02d}:00 in {args.timezone}."
    )
    return 0


def is_intended_local_hour(
    event_name: str,
    schedule: str,
    target_hour: int,
    timezone_name: str,
    utc_date: date | None = None,
) -> bool:
    """Return true when a workflow event should run for the target local hour."""
    if event_name == "workflow_dispatch":
        return True
    if event_name != "schedule":
        return False

    minute, hour = parse_cron_minute_hour(schedule)
    scheduled_date = utc_date or datetime.now(timezone.utc).date()
    scheduled_utc = datetime(
        scheduled_date.year,
        scheduled_date.month,
        scheduled_date.day,
        hour,
        minute,
        tzinfo=timezone.utc,
    )
    scheduled_local = scheduled_utc.astimezone(ZoneInfo(timezone_name))
    return scheduled_local.hour == target_hour


def parse_cron_minute_hour(schedule: str) -> tuple[int, int]:
    """Return the minute and hour from a simple GitHub cron expression."""
    fields = schedule.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 cron fields, got {len(fields)}: {schedule!r}")
    minute = parse_single_int(fields[0], "minute")
    hour = parse_single_int(fields[1], "hour")
    return minute, hour


def parse_single_int(value: str, field_name: str) -> int:
    """Parse cron fields that are intentionally single numeric values."""
    if not value.isdigit():
        raise ValueError(f"Expected a single numeric {field_name}, got {value!r}")
    return int(value)


def parse_utc_date(value: str | None) -> date | None:
    """Parse an optional YYYY-MM-DD date used by tests."""
    if not value:
        return None
    return date.fromisoformat(value)


if __name__ == "__main__":
    raise SystemExit(main())
