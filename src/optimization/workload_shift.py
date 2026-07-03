"""Carbon-aware workload shifting optimizer."""

from __future__ import annotations


def optimize_workload_shift(
    forecast_rows: list[dict[str, object]],
    max_shift_hours: int = 6,
) -> dict[str, object]:
    """Choose a workload shift plan from forecasted price and carbon signals."""
    return {
        "max_shift_hours": max_shift_hours,
        "input_rows": len(forecast_rows),
        "recommended_shift_hours": 0,
    }

