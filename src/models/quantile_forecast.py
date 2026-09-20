"""Quantile forecast calibration and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_NAMES = ("q10", "q50", "q90")
NOMINAL_INTERVAL_COVERAGE = 0.8
MIN_CALIBRATION_ROWS = 24


def residual_quantile_offsets(
    actuals: np.ndarray | pd.Series,
    point_predictions: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return ordered residual offsets for the production quantiles."""
    actual = np.asarray(actuals, dtype=float)
    point = np.asarray(point_predictions, dtype=float)
    residuals = actual - point
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return {name: 0.0 for name in QUANTILE_NAMES}
    values = np.quantile(residuals, QUANTILES)
    return {name: float(value) for name, value in zip(QUANTILE_NAMES, values)}


def apply_residual_quantiles(
    point_predictions: np.ndarray | pd.Series,
    offsets: dict[str, float],
) -> np.ndarray:
    """Convert point predictions and residual offsets into ordered quantiles."""
    point = np.asarray(point_predictions, dtype=float)
    values = np.column_stack([point + float(offsets[name]) for name in QUANTILE_NAMES])
    return np.maximum.accumulate(values, axis=1)


def add_expanding_quantiles(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    actual_column: str,
    point_column: str,
    output_prefix: str,
    min_calibration_rows: int = MIN_CALIBRATION_ROWS,
) -> pd.DataFrame:
    """Add prequential quantiles using only residuals observed on earlier rows."""
    output = frame.copy()
    quantile_columns = [f"{output_prefix}_{name}" for name in QUANTILE_NAMES]
    for column in quantile_columns:
        output[column] = np.nan
    output["quantile_calibration_rows"] = 0

    for _, indices in output.groupby(group_columns, sort=False, observed=True).groups.items():
        ordered_indices = output.loc[indices].sort_values("timestamp_utc").index
        residual_history: list[float] = []
        for index in ordered_indices:
            point = float(output.at[index, point_column])
            if len(residual_history) >= min_calibration_rows:
                offsets = residual_quantile_offsets(
                    np.asarray(residual_history, dtype=float),
                    np.zeros(len(residual_history), dtype=float),
                )
                quantiles = apply_residual_quantiles(np.asarray([point]), offsets)[0]
                for column, value in zip(quantile_columns, quantiles):
                    output.at[index, column] = value
                output.at[index, "quantile_calibration_rows"] = len(residual_history)
            actual = float(output.at[index, actual_column])
            if np.isfinite(actual) and np.isfinite(point):
                residual_history.append(actual - point)

    return output


def quantile_metrics(
    actuals: np.ndarray | pd.Series,
    q10: np.ndarray | pd.Series,
    q50: np.ndarray | pd.Series,
    q90: np.ndarray | pd.Series,
) -> dict[str, float | int | None]:
    """Calculate proper scores and calibration diagnostics for an 80% interval."""
    actual = np.asarray(actuals, dtype=float)
    predictions = np.column_stack(
        [np.asarray(q10, dtype=float), np.asarray(q50, dtype=float), np.asarray(q90, dtype=float)]
    )
    valid = np.isfinite(actual) & np.isfinite(predictions).all(axis=1)
    if not valid.any():
        return empty_quantile_metrics()

    actual = actual[valid]
    predictions = np.maximum.accumulate(predictions[valid], axis=1)
    losses: list[float] = []
    result: dict[str, float | int | None] = {"quantile_rows": int(len(actual))}
    for index, (quantile, name) in enumerate(zip(QUANTILES, QUANTILE_NAMES)):
        residual = actual - predictions[:, index]
        loss = np.maximum(quantile * residual, (quantile - 1) * residual)
        value = float(np.mean(loss))
        result[f"pinball_loss_{name}"] = value
        losses.append(value)

    lower = predictions[:, 0]
    upper = predictions[:, 2]
    width = upper - lower
    below = actual < lower
    above = actual > upper
    coverage = float((~below & ~above).mean())
    alpha = 1.0 - NOMINAL_INTERVAL_COVERAGE
    winkler = width + (2.0 / alpha) * (lower - actual) * below + (2.0 / alpha) * (
        actual - upper
    ) * above
    scale = float(np.mean(np.abs(actual)))
    result.update(
        {
            "mean_pinball_loss": float(np.mean(losses)),
            "interval_coverage_80": coverage,
            "interval_coverage_error_80": coverage - NOMINAL_INTERVAL_COVERAGE,
            "mean_interval_width_80": float(np.mean(width)),
            "median_interval_width_80": float(np.median(width)),
            "normalized_interval_width_80": float(np.mean(width) / scale) if scale > 0 else None,
            "below_interval_rate_80": float(below.mean()),
            "above_interval_rate_80": float(above.mean()),
            "winkler_interval_score_80": float(np.mean(winkler)),
        }
    )
    return result


def empty_quantile_metrics() -> dict[str, float | int | None]:
    """Return a stable unavailable metric schema."""
    return {
        "quantile_rows": 0,
        "pinball_loss_q10": None,
        "pinball_loss_q50": None,
        "pinball_loss_q90": None,
        "mean_pinball_loss": None,
        "interval_coverage_80": None,
        "interval_coverage_error_80": None,
        "mean_interval_width_80": None,
        "median_interval_width_80": None,
        "normalized_interval_width_80": None,
        "below_interval_rate_80": None,
        "above_interval_rate_80": None,
        "winkler_interval_score_80": None,
    }


def build_quantile_artifact(
    base_model: Any,
    actuals: np.ndarray | pd.Series,
    point_predictions: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Package a point estimator with out-of-sample residual quantiles."""
    return {
        "artifact_type": "empirical_residual_quantile_forecast",
        "quantiles": list(QUANTILES),
        "nominal_interval_coverage": NOMINAL_INTERVAL_COVERAGE,
        "calibration_method": "walk_forward_out_of_sample_residual_quantiles",
        "guaranteed_coverage": False,
        "calibration_rows": int(
            np.isfinite(np.asarray(actuals, dtype=float) - np.asarray(point_predictions, dtype=float)).sum()
        ),
        "residual_offsets": residual_quantile_offsets(actuals, point_predictions),
        "base_model": base_model,
    }


def is_quantile_artifact(model: Any) -> bool:
    """Return whether a loaded object is a production quantile artifact."""
    return isinstance(model, dict) and model.get("artifact_type") == (
        "empirical_residual_quantile_forecast"
    )


def predict_quantile_artifact(model: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Predict q10, q50, and q90 from a persisted quantile artifact."""
    if not is_quantile_artifact(model):
        raise ValueError("Expected a calibrated quantile forecast artifact")
    point = model["base_model"].predict(features)
    return apply_residual_quantiles(point, model["residual_offsets"])
