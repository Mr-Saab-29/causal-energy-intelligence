"""Causal effect estimation utilities."""

from __future__ import annotations


def estimate_treatment_effect(
    treatment: str,
    outcome: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Estimate the causal effect of a treatment on an outcome."""
    return {
        "treatment": treatment,
        "outcome": outcome,
        "effect": None,
        "rows": len(records),
    }

