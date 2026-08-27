"""Decide whether GitHub started the scheduled workflow in the Paris run window."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> int:
    """Write a GitHub Actions should_run output for the daily schedule guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--schedule", default="")
    parser.add_argument("--window-start-hour", type=int, default=0)
    parser.add_argument("--window-end-hour", type=int, default=6)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--now-utc", default=None)
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args()

    should_run = is_local_time_in_window(
        event_name=args.event_name,
        window_start_hour=args.window_start_hour,
        window_end_hour=args.window_end_hour,
        timezone_name=args.timezone,
        now_utc=parse_utc_datetime(args.now_utc),
    )
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"should_run={str(should_run).lower()}\n")
    print(
        "Running scheduled workflow."
        if should_run
        else (
            "Skipping because the runner started outside "
            f"{args.window_start_hour:02d}:00-{args.window_end_hour:02d}:00 "
            f"in {args.timezone}."
        )
    )
    return 0


def is_local_time_in_window(
    event_name: str,
    window_start_hour: int,
    window_end_hour: int,
    timezone_name: str,
    now_utc: datetime | None = None,
) -> bool:
    """Return true when a workflow event should run in the local time window."""
    if event_name == "workflow_dispatch":
        return True
    if event_name != "schedule":
        return False
    if not 0 <= window_start_hour <= 23:
        raise ValueError("window_start_hour must be between 0 and 23")
    if not 1 <= window_end_hour <= 24:
        raise ValueError("window_end_hour must be between 1 and 24")
    if window_start_hour >= window_end_hour:
        raise ValueError("window_start_hour must be before window_end_hour")

    current_utc = now_utc or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    current_local = current_utc.astimezone(ZoneInfo(timezone_name))
    return window_start_hour <= current_local.hour < window_end_hour


def parse_utc_datetime(value: str | None) -> datetime | None:
    """Parse an optional UTC datetime used by tests."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
