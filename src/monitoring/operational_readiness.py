"""Readiness checks for deployed operational recommendation outcome audits."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.baseline_price import TIMESTAMP_COLUMN

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data/processed/modeling_price_features.csv"
OPERATIONAL_RECOMMENDATION_HISTORY_PATH = (
    ROOT / "reports/monitoring/operational_recommendation_history.csv"
)
OUTPUT_PATH = ROOT / "reports/metrics/operational_audit_readiness.json"


def build_operational_audit_readiness(
    features_path: str | Path = FEATURES_PATH,
    recommendation_history_path: str | Path = OPERATIONAL_RECOMMENDATION_HISTORY_PATH,
    output_path: str | Path = OUTPUT_PATH,
    as_of_utc: str | pd.Timestamp | None = None,
    max_actual_lag_days: int = 7,
    require_current_month_recommendations: bool = True,
) -> dict[str, Any]:
    """Verify actuals and recommendation history are available for current-month audit."""
    as_of = normalize_as_of(as_of_utc)
    month_start = as_of.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    features = read_csv(features_path)
    recommendations = read_csv(recommendation_history_path)
    checks: dict[str, Any] = {
        "current_month_actuals_available": False,
        "current_month_recommendations_available": False,
        "settled_current_month_recommendations_available": False,
    }
    reasons: list[str] = []
    warnings: list[str] = []

    latest_actual = None
    if features.empty or TIMESTAMP_COLUMN not in features:
        reasons.append("modeling_price_features_missing")
        current_month_actual_rows = 0
    else:
        features = normalize_timestamp_column(features)
        latest_actual = features[TIMESTAMP_COLUMN].max()
        current_month_actual_rows = int((features[TIMESTAMP_COLUMN] >= month_start).sum())
        checks["current_month_actuals_available"] = current_month_actual_rows > 0
        if current_month_actual_rows == 0:
            reasons.append("current_month_actuals_missing")
        actual_lag_days = (
            (as_of - latest_actual).total_seconds() / 86_400
            if latest_actual is not None and not pd.isna(latest_actual)
            else None
        )
        if actual_lag_days is not None and actual_lag_days > max_actual_lag_days:
            reasons.append("latest_actuals_too_stale")

    current_month_recommendation_rows = 0
    settled_current_month_recommendation_rows = 0
    if recommendations.empty or TIMESTAMP_COLUMN not in recommendations:
        if require_current_month_recommendations:
            reasons.append("operational_recommendation_history_missing")
    else:
        recommendations = normalize_timestamp_column(recommendations)
        decision_dates = decision_group_dates(recommendations)
        current_month_mask = decision_dates >= month_start
        current_month_recommendation_rows = int(current_month_mask.sum())
        checks["current_month_recommendations_available"] = (
            current_month_recommendation_rows > 0
        )
        if current_month_recommendation_rows == 0 and require_current_month_recommendations:
            reasons.append("current_month_operational_recommendations_missing")
        if latest_actual is not None and not pd.isna(latest_actual):
            settled_current_month_recommendation_rows = int(
                (current_month_mask & (recommendations[TIMESTAMP_COLUMN] <= latest_actual)).sum()
            )
            checks["settled_current_month_recommendations_available"] = (
                settled_current_month_recommendation_rows > 0
            )
            if current_month_recommendation_rows > 0 and settled_current_month_recommendation_rows == 0:
                reasons.append("current_month_recommendations_not_settled_yet")

    status = "fail" if reasons else "warn" if warnings else "pass"
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "as_of_utc": as_of.isoformat(),
        "month_start_utc": month_start.isoformat(),
        "latest_actual_timestamp_utc": latest_actual.isoformat()
        if latest_actual is not None and not pd.isna(latest_actual)
        else None,
        "max_actual_lag_days": max_actual_lag_days,
        "current_month_actual_rows": current_month_actual_rows,
        "current_month_recommendation_rows": current_month_recommendation_rows,
        "settled_current_month_recommendation_rows": settled_current_month_recommendation_rows,
        "checks": checks,
        "reasons": reasons,
        "warnings": warnings,
    }
    write_json(output_path, report)
    return report


def normalize_as_of(value: str | pd.Timestamp | None) -> pd.Timestamp:
    """Return timezone-aware UTC timestamp for readiness checks."""
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp_utc and drop invalid rows."""
    output = frame.copy()
    output[TIMESTAMP_COLUMN] = pd.to_datetime(output[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    return output.dropna(subset=[TIMESTAMP_COLUMN])


def decision_group_dates(frame: pd.DataFrame) -> pd.Series:
    """Return UTC decision-group dates, falling back to timestamp dates."""
    if "decision_group" not in frame:
        return frame[TIMESTAMP_COLUMN].dt.floor("D")
    parsed = pd.to_datetime(frame["decision_group"], utc=True, errors="coerce")
    fallback = frame[TIMESTAMP_COLUMN].dt.floor("D")
    return parsed.fillna(fallback)


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV if it exists."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run operational audit readiness checks from the command line."""
    parser = argparse.ArgumentParser(description="Verify current-month audit readiness.")
    parser.add_argument("--as-of-utc", default=None)
    parser.add_argument("--max-actual-lag-days", type=int, default=7)
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--allow-missing-current-month-recommendations",
        action="store_true",
    )
    args = parser.parse_args(argv)
    report = build_operational_audit_readiness(
        as_of_utc=args.as_of_utc,
        max_actual_lag_days=args.max_actual_lag_days,
        output_path=args.output_path,
        require_current_month_recommendations=not args.allow_missing_current_month_recommendations,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "reasons": report["reasons"],
                "warnings": report["warnings"],
                "latest_actual_timestamp_utc": report["latest_actual_timestamp_utc"],
                "current_month_actual_rows": report["current_month_actual_rows"],
                "current_month_recommendation_rows": report[
                    "current_month_recommendation_rows"
                ],
                "settled_current_month_recommendation_rows": report[
                    "settled_current_month_recommendation_rows"
                ],
            },
            indent=2,
        )
    )
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
