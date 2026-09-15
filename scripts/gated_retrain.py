"""Run retraining behind an incumbent-vs-candidate promotion gate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION_PATH = ROOT / "reports/metrics/model_promotion_decision.json"
CHAMPION_PATH = ROOT / "reports/metrics/champion_model_selection.json"
OPERATIONAL_OUTCOME_AUDIT_PATH = ROOT / "reports/metrics/recommendation_outcome_audit.json"
SNAPSHOT_ROOT = ROOT / ".tmp/retrain_snapshots"
MAX_GUARDED_METRIC_DEGRADATION = 1.05
MIN_OPERATIONAL_DECISION_GROUPS = 7
MIN_OPERATIONAL_TOP_5_HIT_RATE = 0.6
MAX_OPERATIONAL_CARBON_REGRET_G_CO2E_PER_KWH = 2.0

SNAPSHOT_PATHS = [
    ROOT / "models",
    ROOT / "reports/metrics",
    ROOT / "reports/predictions",
    ROOT / "reports/rankings",
    ROOT / "reports/recommendations",
    ROOT / "reports/scenarios",
    ROOT / "reports/carbon",
    ROOT / "frontend/public/data/dashboard.json",
]

PROMOTION_METRICS = {
    "recommendation_regret": "mean_top_1_combined_regret",
    "carbon_intensity_error": "carbon_intensity_mae_g_co2e_per_kwh",
    "carbon_regret": "carbon_regret_g_co2e_per_kwh",
    "top_5_ranking_loss": "top_5_ranking_loss",
    "price_direction_error": "price_direction_error",
}


@dataclass(frozen=True)
class SnapshotEntry:
    """One snapshotted production artifact path."""

    source: Path
    snapshot: Path
    existed: bool
    is_dir: bool


def main(argv: list[str] | None = None) -> int:
    """Run candidate retraining and promote only when it beats the incumbent."""
    parser = argparse.ArgumentParser(description="Gate model retraining promotion.")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Required relative improvement over incumbent weighted score.",
    )
    parser.add_argument(
        "--decision-path",
        default=str(DEFAULT_DECISION_PATH),
        help="JSON path for the promotion decision report.",
    )
    parser.add_argument(
        "--keep-snapshots",
        type=int,
        default=3,
        help="Number of latest retrain snapshots to keep.",
    )
    parser.add_argument(
        "--operational-outcome-path",
        default=str(OPERATIONAL_OUTCOME_AUDIT_PATH),
        help="Settled operational outcome metrics used as promotion evidence.",
    )
    parser.add_argument(
        "--min-operational-decision-groups",
        type=int,
        default=MIN_OPERATIONAL_DECISION_GROUPS,
        help="Minimum settled decision days before operational metrics can block promotion.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Candidate retraining command. Prefix with -- before the command.",
    )
    args = parser.parse_args(argv)
    command = normalize_command(args.command)
    if not command:
        parser.error("candidate command is required, for example: -- make forecast-all-candidate")

    incumbent = load_json(CHAMPION_PATH)
    snapshot_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_dir = SNAPSHOT_ROOT / snapshot_id
    snapshot_entries = create_snapshot(snapshot_dir)
    command_result = run_command(command)

    if command_result.returncode != 0:
        restore_snapshot(snapshot_entries)
        decision = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "status": "candidate_failed_restored_incumbent",
            "promoted": False,
            "command": command,
            "returncode": command_result.returncode,
            "stdout_tail": (command_result.stdout or "")[-4000:],
            "stderr_tail": (command_result.stderr or "")[-4000:],
        }
        write_json(Path(args.decision_path), decision)
        prune_snapshots(SNAPSHOT_ROOT, keep=args.keep_snapshots)
        print(json.dumps(decision, indent=2))
        return command_result.returncode

    candidate = load_json(CHAMPION_PATH)
    operational_outcome = load_json(Path(args.operational_outcome_path))
    decision = evaluate_promotion(
        incumbent=incumbent,
        candidate=candidate,
        min_improvement=args.min_improvement,
        operational_outcome=operational_outcome,
        min_operational_decision_groups=args.min_operational_decision_groups,
    )
    decision.update(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "snapshot_dir": str(snapshot_dir.relative_to(ROOT)),
        }
    )

    if not decision["promoted"]:
        restore_snapshot(snapshot_entries)
        decision["status"] = "candidate_rejected_restored_incumbent"
    else:
        decision["status"] = "candidate_promoted"

    write_json(Path(args.decision_path), decision)
    prune_snapshots(SNAPSHOT_ROOT, keep=args.keep_snapshots)
    print(json.dumps(decision, indent=2))
    return 0


def normalize_command(command: list[str]) -> list[str]:
    """Remove argparse's separator marker from the candidate command."""
    if command and command[0] == "--":
        return command[1:]
    return command


def create_snapshot(snapshot_dir: Path) -> list[SnapshotEntry]:
    """Snapshot current production model/dashboard artifacts."""
    entries: list[SnapshotEntry] = []
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source in SNAPSHOT_PATHS:
        relative = source.relative_to(ROOT)
        target = snapshot_dir / relative
        existed = source.exists()
        is_dir = source.is_dir()
        entries.append(SnapshotEntry(source=source, snapshot=target, existed=existed, is_dir=is_dir))
        if not existed:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return entries


def restore_snapshot(entries: list[SnapshotEntry]) -> None:
    """Restore production artifacts from a snapshot."""
    for entry in entries:
        if entry.source.exists():
            if entry.source.is_dir():
                shutil.rmtree(entry.source)
            else:
                entry.source.unlink()
        if not entry.existed:
            continue
        entry.source.parent.mkdir(parents=True, exist_ok=True)
        if entry.is_dir:
            shutil.copytree(entry.snapshot, entry.source)
        else:
            shutil.copy2(entry.snapshot, entry.source)


def prune_snapshots(snapshot_root: Path, keep: int) -> None:
    """Keep only the newest retrain snapshots."""
    if keep < 1 or not snapshot_root.exists():
        return
    snapshots = sorted(
        (path for path in snapshot_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots[keep:]:
        shutil.rmtree(snapshot)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the candidate retraining command."""
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        check=False,
    )


def evaluate_promotion(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
    min_improvement: float,
    operational_outcome: dict[str, Any] | None = None,
    min_operational_decision_groups: int = MIN_OPERATIONAL_DECISION_GROUPS,
) -> dict[str, Any]:
    """Compare candidate champion against incumbent champion."""
    incumbent_row = selected_model_row(incumbent)
    candidate_row = selected_model_row(candidate)
    if not incumbent_row:
        return {
            "promoted": True,
            "reason": "no incumbent champion metrics were available",
            "incumbent": summarize_champion(incumbent, incumbent_row),
            "candidate": summarize_champion(candidate, candidate_row),
        }
    if not candidate_row:
        return {
            "promoted": False,
            "reason": "candidate champion metrics were unavailable",
            "incumbent": summarize_champion(incumbent, incumbent_row),
            "candidate": summarize_champion(candidate, candidate_row),
        }

    weights = incumbent.get("weights") or candidate.get("weights") or {}
    weighted_ratios: dict[str, float] = {}
    weighted_score = 0.0
    weight_sum = 0.0
    for weight_name, metric_name in PROMOTION_METRICS.items():
        weight = float(weights.get(weight_name, 0.0))
        if weight <= 0:
            continue
        incumbent_value = as_float(incumbent_row.get(metric_name))
        candidate_value = as_float(candidate_row.get(metric_name))
        ratio = lower_is_better_ratio(candidate_value, incumbent_value)
        weighted_ratios[metric_name] = ratio
        weighted_score += weight * ratio
        weight_sum += weight
    promotion_score = weighted_score / weight_sum if weight_sum else float("inf")
    required_score = 1.0 - min_improvement
    guarded_degradations = {
        metric: ratio
        for metric, ratio in weighted_ratios.items()
        if metric
        in {
            "mean_top_1_combined_regret",
            "carbon_regret_g_co2e_per_kwh",
        }
        and ratio > MAX_GUARDED_METRIC_DEGRADATION
    }
    operational_evidence = evaluate_operational_evidence(
        operational_outcome or {},
        min_decision_groups=min_operational_decision_groups,
    )
    promoted = (
        promotion_score < required_score
        and not guarded_degradations
        and not operational_evidence["blocks_promotion"]
    )
    reason = promotion_reason(
        promoted=promoted,
        offline_score_passed=promotion_score < required_score,
        guarded_degradations=guarded_degradations,
        operational_evidence=operational_evidence,
    )
    return {
        "promoted": promoted,
        "reason": reason,
        "promotion_score_vs_incumbent": round(promotion_score, 6),
        "required_score": round(required_score, 6),
        "weighted_metric_ratios": {
            metric: round(value, 6) for metric, value in weighted_ratios.items()
        },
        "guarded_metric_degradation_limit": MAX_GUARDED_METRIC_DEGRADATION,
        "guarded_metric_degradations": {
            metric: round(value, 6) for metric, value in guarded_degradations.items()
        },
        "operational_evidence": operational_evidence,
        "incumbent": summarize_champion(incumbent, incumbent_row),
        "candidate": summarize_champion(candidate, candidate_row),
    }


def promotion_reason(
    promoted: bool,
    offline_score_passed: bool,
    guarded_degradations: dict[str, float],
    operational_evidence: dict[str, Any],
) -> str:
    """Return the primary promotion decision reason."""
    if promoted:
        return "candidate weighted relative score improved"
    if not offline_score_passed:
        return "candidate did not beat incumbent weighted relative score"
    if guarded_degradations:
        return "candidate regressed on guarded recommendation metrics"
    if operational_evidence["blocks_promotion"]:
        return "settled operational metrics below promotion floor"
    return "candidate did not beat incumbent weighted relative score"


def evaluate_operational_evidence(
    report: dict[str, Any],
    min_decision_groups: int = MIN_OPERATIONAL_DECISION_GROUPS,
) -> dict[str, Any]:
    """Summarize settled operational outcomes for the promotion decision."""
    if not report or not report.get("available"):
        return {
            "status": "unavailable",
            "blocks_promotion": False,
            "reason": report.get("reason", "recommendation outcome audit unavailable")
            if report
            else "recommendation outcome audit unavailable",
            "minimum_decision_groups": min_decision_groups,
        }

    decision_groups = int(report.get("decision_groups", 0) or 0)
    top_5_hit_rate = as_optional_float(report.get("top_5_hit_rate"))
    carbon_regret = as_optional_float(report.get("mean_carbon_regret_g_co2e_per_kwh"))
    evidence = {
        "status": "insufficient_history"
        if decision_groups < min_decision_groups
        else "guarded",
        "blocks_promotion": False,
        "decision_groups": decision_groups,
        "rows": int(report.get("rows", 0) or 0),
        "minimum_decision_groups": min_decision_groups,
        "top_1_hit_rate": as_optional_float(report.get("top_1_hit_rate")),
        "top_5_hit_rate": top_5_hit_rate,
        "mean_actual_rank_of_top_1": as_optional_float(
            report.get("mean_actual_rank_of_top_1")
        ),
        "mean_carbon_regret_g_co2e_per_kwh": carbon_regret,
        "floors": {
            "min_top_5_hit_rate": MIN_OPERATIONAL_TOP_5_HIT_RATE,
            "max_mean_carbon_regret_g_co2e_per_kwh": (
                MAX_OPERATIONAL_CARBON_REGRET_G_CO2E_PER_KWH
            ),
        },
        "failures": [],
    }
    if decision_groups < min_decision_groups:
        evidence["reason"] = "not enough settled operational decision days"
        return evidence

    failures: list[str] = []
    if top_5_hit_rate is not None and top_5_hit_rate < MIN_OPERATIONAL_TOP_5_HIT_RATE:
        failures.append("top_5_hit_rate_below_floor")
    if (
        carbon_regret is not None
        and carbon_regret > MAX_OPERATIONAL_CARBON_REGRET_G_CO2E_PER_KWH
    ):
        failures.append("carbon_regret_above_floor")
    evidence["failures"] = failures
    evidence["blocks_promotion"] = bool(failures)
    evidence["status"] = "guarded_fail" if failures else "guarded_pass"
    return evidence


def selected_model_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the metrics row for a payload's selected champion model."""
    champion_model = payload.get("champion_model")
    for row in payload.get("models", []):
        if row.get("model") == champion_model:
            return row
    return {}


def summarize_champion(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Return compact champion fields for the promotion report."""
    return {
        "champion_model": payload.get("champion_model"),
        "metrics": {
            metric_name: row.get(metric_name)
            for metric_name in PROMOTION_METRICS.values()
        },
    }


def lower_is_better_ratio(candidate: float, incumbent: float) -> float:
    """Return candidate/incumbent for lower-is-better metrics."""
    if incumbent == 0:
        return 0.0 if candidate <= 0 else float("inf")
    return candidate / incumbent


def as_float(value: Any) -> float:
    """Parse a metric value as float, using infinity for missing values."""
    if value is None:
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def as_optional_float(value: Any) -> float | None:
    """Parse an optional metric value as float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON if present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
