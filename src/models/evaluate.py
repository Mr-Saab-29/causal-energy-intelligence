"""Forecast model evaluation utilities."""

from __future__ import annotations


def evaluate_forecast(predictions: list[float], actuals: list[float]) -> dict[str, float]:
    """Evaluate forecast predictions against observed values."""
    if not predictions or not actuals:
        return {"mae": 0.0}

    paired_values = zip(predictions, actuals, strict=False)
    errors = [abs(prediction - actual) for prediction, actual in paired_values]
    return {"mae": sum(errors) / len(errors)}

