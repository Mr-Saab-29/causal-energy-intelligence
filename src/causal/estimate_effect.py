"""Causal effect estimation utilities."""

from __future__ import annotations

from typing import Any

from src.causal.estimand import load_estimand_spec, validate_estimand_spec


def estimate_treatment_effect(
    treatment: str,
    outcome: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Validate an effect request against the contract before an estimator exists."""
    spec: dict[str, Any] = load_estimand_spec()
    errors = validate_estimand_spec(spec)
    if errors:
        raise ValueError(f"Invalid causal estimand contract: {', '.join(errors)}")
    expected_treatment = spec["treatment"]["graph_node"]
    expected_outcome = spec["outcome"]["graph_node"]
    if treatment != expected_treatment or outcome != expected_outcome:
        raise ValueError(
            "Treatment-effect request does not match "
            f"{spec['estimand_version']}: expected {expected_treatment} -> {expected_outcome}."
        )
    return {
        "treatment": treatment,
        "outcome": outcome,
        "effect": None,
        "rows": len(records),
        "status": "not_estimated",
        "reason": "identified_estimator_not_implemented",
        "estimand_version": spec["estimand_version"],
    }
