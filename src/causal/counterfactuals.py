"""Counterfactual scenario utilities."""

from __future__ import annotations


def simulate_counterfactual(
    scenario: dict[str, object],
    baseline: list[dict[str, object]],
) -> dict[str, object]:
    """Simulate a counterfactual energy scenario against a baseline."""
    return {"scenario": scenario, "baseline_rows": len(baseline), "result": None}

