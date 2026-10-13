"""Operational regret and estimator-drift monitoring for causal recommendations."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "reports/monitoring/recommendation_outcome_audit.csv"
ESTIMATOR_PATH = ROOT / "reports/metrics/causal_identified_estimator.json"
DML_PATH = ROOT / "reports/metrics/causal_dml_estimator.json"
HISTORY_PATH = ROOT / "reports/monitoring/causal_estimator_history.csv"
OUTPUT_PATH = ROOT / "reports/metrics/causal_performance_monitoring.json"
MIN_SETTLED_GROUPS = 14
MAX_MEAN_CARBON_REGRET = 10.0
MAX_RELATIVE_ESTIMATOR_DRIFT = 0.30
HISTORY_COLUMNS = [
    "generated_at_utc",
    "estimator",
    "horizon_hours",
    "estimate_kgco2e_per_mwh",
]


def build_causal_performance_report(
    audit: pd.DataFrame,
    estimator_reports: list[tuple[str, dict[str, Any]]],
    history: pd.DataFrame | None = None,
    generated_at_utc: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize settled causal regret and changes in identified estimates."""
    generated_at_utc = generated_at_utc or datetime.now(UTC).isoformat()
    realized = summarize_realized_regret(audit)
    snapshots = estimator_snapshots(estimator_reports, generated_at_utc)
    combined_history = append_snapshot_history(history, snapshots)
    drift = summarize_estimator_drift(combined_history, snapshots)
    statuses = {realized["status"], drift["status"]}
    status = "pass" if statuses == {"pass"} else "fail" if "fail" in statuses else "insufficient_history"
    return (
        {
            "generated_at_utc": generated_at_utc,
            "status": status,
            "realized_causal_regret": realized,
            "estimator_drift": drift,
        },
        combined_history,
    )


def summarize_realized_regret(audit: pd.DataFrame) -> dict[str, Any]:
    if audit.empty or "recommendation_basis" not in audit:
        return {"status": "insufficient_history", "settled_decision_groups": 0}
    causal = audit[
        audit["recommendation_basis"].astype(str).str.startswith("causal_adjusted")
    ].copy()
    if "recommendation_rank" in causal:
        causal = causal[pd.to_numeric(causal["recommendation_rank"], errors="coerce") == 1]
    groups = causal.get("decision_group", pd.Series(dtype=object)).nunique()
    regret = safe_mean(causal.get("carbon_regret_g_co2e_per_kwh_observed"))
    if groups < MIN_SETTLED_GROUPS or regret is None:
        status = "insufficient_history"
    else:
        status = "pass" if regret <= MAX_MEAN_CARBON_REGRET else "fail"
    return {
        "status": status,
        "settled_decision_groups": int(groups),
        "minimum_settled_decision_groups": MIN_SETTLED_GROUPS,
        "mean_carbon_regret_g_co2e_per_kwh": regret,
        "maximum_mean_carbon_regret_g_co2e_per_kwh": MAX_MEAN_CARBON_REGRET,
    }


def estimator_snapshots(
    reports: list[tuple[str, dict[str, Any]]], generated_at_utc: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for estimator_name, report in reports:
        if report.get("status") != "ok":
            continue
        for horizon in report.get("horizons") or []:
            points = horizon.get("point_estimates") or {}
            estimate = points.get("orthogonal_mean_marginal_response_kgco2e_per_mwh")
            if estimate is None:
                estimate = points.get("mean_marginal_response_kgco2e_per_mwh")
            if estimate is None:
                continue
            rows.append(
                {
                    "generated_at_utc": generated_at_utc,
                    "estimator": estimator_name,
                    "horizon_hours": int(horizon["horizon_hours"]),
                    "estimate_kgco2e_per_mwh": float(estimate),
                }
            )
    return pd.DataFrame(rows)


def append_snapshot_history(history: pd.DataFrame | None, snapshots: pd.DataFrame) -> pd.DataFrame:
    existing = history.copy() if history is not None else pd.DataFrame(columns=HISTORY_COLUMNS)
    if snapshots.empty:
        return existing.reindex(columns=HISTORY_COLUMNS)
    combined = pd.concat([existing, snapshots], ignore_index=True)
    return combined.drop_duplicates(
        subset=["generated_at_utc", "estimator", "horizon_hours"], keep="last"
    )


def summarize_estimator_drift(history: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
    if snapshots.empty:
        return {"status": "insufficient_history", "comparisons": []}
    comparisons: list[dict[str, Any]] = []
    for row in snapshots.itertuples(index=False):
        prior = history[
            (history["estimator"] == row.estimator)
            & (pd.to_numeric(history["horizon_hours"], errors="coerce") == row.horizon_hours)
            & (history["generated_at_utc"] != row.generated_at_utc)
        ]
        if prior.empty:
            continue
        previous = float(prior.iloc[-1]["estimate_kgco2e_per_mwh"])
        relative = abs(float(row.estimate_kgco2e_per_mwh) - previous) / max(abs(previous), 1.0)
        comparisons.append(
            {
                "estimator": row.estimator,
                "horizon_hours": int(row.horizon_hours),
                "previous_estimate": previous,
                "current_estimate": float(row.estimate_kgco2e_per_mwh),
                "relative_drift": round(relative, 4),
                "status": "pass" if relative <= MAX_RELATIVE_ESTIMATOR_DRIFT else "fail",
            }
        )
    if not comparisons:
        return {"status": "insufficient_history", "comparisons": []}
    return {
        "status": "fail" if any(row["status"] == "fail" for row in comparisons) else "pass",
        "maximum_relative_drift": MAX_RELATIVE_ESTIMATOR_DRIFT,
        "comparisons": comparisons,
    }


def safe_mean(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    value = pd.to_numeric(values, errors="coerce").mean()
    return None if pd.isna(value) else round(float(value), 4)


def read_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monitor causal regret and estimator drift.")
    parser.add_argument("--audit-path", default=str(AUDIT_PATH))
    parser.add_argument("--history-path", default=str(HISTORY_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    audit = pd.read_csv(args.audit_path) if Path(args.audit_path).exists() else pd.DataFrame()
    try:
        history = (
            pd.read_csv(args.history_path)
            if Path(args.history_path).exists()
            else pd.DataFrame(columns=HISTORY_COLUMNS)
        )
    except pd.errors.EmptyDataError:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)
    report, history = build_causal_performance_report(
        audit,
        [("constrained_regression", read_json(ESTIMATOR_PATH)), ("double_machine_learning", read_json(DML_PATH))],
        history,
    )
    Path(args.history_path).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history_path, index=False)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
