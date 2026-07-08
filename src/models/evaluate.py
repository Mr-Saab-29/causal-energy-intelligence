"""Forecast model evaluation utilities."""

from __future__ import annotations

import numpy as np


def evaluate_forecast(predictions: list[float], actuals: list[float]) -> dict[str, float]:
    """Evaluate forecast predictions against observed values."""
    if not predictions or not actuals:
        return {"mae": 0.0}

    paired_values = zip(predictions, actuals, strict=False)
    errors = [abs(prediction - actual) for prediction, actual in paired_values]
    return {"mae": sum(errors) / len(errors)}


def evaluate_regression_forecast(predictions: list[float], actuals: list[float]) -> dict[str, float]:
    """Evaluate regression forecasts with common forecasting metrics."""
    prediction_values = np.asarray(predictions, dtype=float)
    actual_values = np.asarray(actuals, dtype=float)
    if len(prediction_values) != len(actual_values):
        raise ValueError("predictions and actuals must have the same length")
    if len(prediction_values) == 0:
        return {"mae": 0.0, "rmse": 0.0, "smape": 0.0}

    errors = prediction_values - actual_values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    smape = np.mean(
        2 * np.abs(errors) / np.maximum(np.abs(actual_values) + np.abs(prediction_values), 1e-9)
    )
    return {"mae": float(mae), "rmse": float(rmse), "smape": float(smape)}
