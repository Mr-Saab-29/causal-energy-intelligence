"""Write a compact GitHub Actions orchestration decision report."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT_PATH = Path("reports/metrics/orchestration_decision.json")


def main(argv: list[str] | None = None) -> int:
    """Write orchestration status for the daily ingestion workflow."""
    parser = argparse.ArgumentParser(description="Write daily orchestration status.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--event-name", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-attempt", default="")
    parser.add_argument("--preflight-result", default="")
    parser.add_argument("--retrain-requested", default="")
    parser.add_argument("--retrain-result", default="")
    parser.add_argument("--operational-action", default="")
    parser.add_argument("--operational-reasons", default="")
    parser.add_argument("--state-source", default="")
    parser.add_argument("--publish-status", default="completed")
    args = parser.parse_args(argv)

    retrain_requested = parse_bool(args.retrain_requested)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "event_name": args.event_name,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "preflight": {
            "result": args.preflight_result,
            "action": args.operational_action,
            "retrain_requested": retrain_requested,
            "reasons": parse_reasons(args.operational_reasons),
        },
        "retrain": {
            "requested": retrain_requested,
            "result": args.retrain_result,
        },
        "publish": {
            "status": args.publish_status,
            "state_source": args.state_source,
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def parse_bool(value: str) -> bool:
    """Parse GitHub-style boolean strings."""
    return value.lower() == "true"


def parse_reasons(value: str) -> list[str]:
    """Parse comma-separated operational action reasons."""
    return [reason for reason in value.split(",") if reason]


if __name__ == "__main__":
    raise SystemExit(main())
