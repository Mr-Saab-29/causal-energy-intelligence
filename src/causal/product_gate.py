"""Evidence gate for promoting causal recommendations in the product."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_PATH = ROOT / "reports/metrics/causal_diagnostics.json"
STRUCTURAL_PATH = ROOT / "reports/metrics/causal_structural_validation.json"
MONITORING_PATH = ROOT / "reports/metrics/causal_performance_monitoring.json"
RECOMMENDATION_METRICS_PATH = ROOT / "reports/metrics/marginal_ranking_shift_metrics.json"
OUTPUT_PATH = ROOT / "reports/metrics/causal_product_gate.json"


def evaluate_product_gate(
    diagnostics: dict[str, Any],
    structural: dict[str, Any],
    monitoring: dict[str, Any],
    recommendation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed product status from identification evidence."""
    horizons = diagnostics.get("horizons") if isinstance(diagnostics.get("horizons"), list) else []
    observed_evidence = diagnostics.get("data_origin") == "observed"
    overlap_checks = [
        horizon.get("treatment_overlap_and_positivity", {}).get("status") == "pass"
        for horizon in horizons
        if isinstance(horizon, dict)
    ]
    uncertainty_checks = [
        horizon.get("time_series_uncertainty", {}).get("status") == "pass"
        for horizon in horizons
        if isinstance(horizon, dict)
    ]
    overlap_ok = bool(overlap_checks) and all(overlap_checks)
    uncertainty_ok = bool(uncertainty_checks) and all(uncertainty_checks)
    diagnostics_ok = bool(
        observed_evidence
        and diagnostics.get("diagnostics_ready")
        and diagnostics.get("causal_claim_eligible")
        and overlap_ok
        and uncertainty_ok
    )
    structural_ok = bool(
        structural.get("structural_validation_ready")
        and structural.get("data_origin") == "pypsa_eur"
        and structural.get("causal_claim_eligible")
        and structural.get("model_risk") == "low"
    )
    monitoring_ok = bool(
        monitoring.get("status") == "pass"
        and monitoring.get("realized_causal_regret", {}).get("status") == "pass"
        and monitoring.get("estimator_drift", {}).get("status") == "pass"
    )
    recommendation_metrics = recommendation_metrics or {}
    future_recommendations = recommendation_metrics.get("future", recommendation_metrics)
    recommendation_engine_ok = bool(
        isinstance(future_recommendations, dict)
        and future_recommendations.get("method") == "identified_causal_effect"
    )
    promoted = diagnostics_ok and structural_ok and monitoring_ok and recommendation_engine_ok

    if promoted:
        evidence_tier = "operationally_validated"
        basis_label = "Causal-adjusted"
    elif diagnostics_ok:
        evidence_tier = "observational_shadow"
        basis_label = "Causal-adjusted MVP"
    else:
        evidence_tier = "marginal_proxy_mvp"
        basis_label = "Causal-adjusted MVP"

    blockers: list[str] = []
    if not diagnostics_ok:
        blockers.append("identified_observational_diagnostics_not_eligible")
    if not structural_ok:
        blockers.append("dispatch_structural_validation_not_eligible")
    if not monitoring_ok:
        blockers.append("operational_causal_monitoring_not_ready")
    if not recommendation_engine_ok:
        blockers.append("recommendation_engine_still_uses_marginal_proxy")

    # Proxy recommendations remain demonstrable before identification, but an
    # identified model must fail closed if its overlap or uncertainty checks fail.
    recommendations_allowed = not observed_evidence or (overlap_ok and uncertainty_ok)
    suppression_reasons: list[str] = []
    if observed_evidence and not overlap_ok:
        suppression_reasons.append("poor_treatment_overlap")
    if observed_evidence and not uncertainty_ok:
        suppression_reasons.append("causal_uncertainty_check_failed")

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "promoted" if promoted else "provisional",
        "promoted": promoted,
        "basis_label": basis_label,
        "evidence_tier": evidence_tier,
        "recommendations_allowed": recommendations_allowed,
        "suppression_reasons": suppression_reasons,
        "blockers": blockers,
        "checks": {
            "observed_diagnostics": diagnostics_ok,
            "treatment_overlap": overlap_ok if observed_evidence else None,
            "time_series_uncertainty": uncertainty_ok if observed_evidence else None,
            "structural_validation": structural_ok,
            "realized_regret_and_drift": monitoring_ok,
            "identified_recommendation_engine": recommendation_engine_ok,
        },
        "disclosure": (
            "Operationally validated causal recommendations."
            if promoted
            else "Provisional marginal-emissions proxy; not validated causal impact."
        ),
    }


def read_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the causal product evidence gate.")
    parser.add_argument("--diagnostics-path", default=str(DIAGNOSTICS_PATH))
    parser.add_argument("--structural-path", default=str(STRUCTURAL_PATH))
    parser.add_argument("--monitoring-path", default=str(MONITORING_PATH))
    parser.add_argument(
        "--recommendation-metrics-path",
        default=str(RECOMMENDATION_METRICS_PATH),
    )
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    report = evaluate_product_gate(
        read_json(args.diagnostics_path),
        read_json(args.structural_path),
        read_json(args.monitoring_path),
        read_json(args.recommendation_metrics_path),
    )
    write_json(args.output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
