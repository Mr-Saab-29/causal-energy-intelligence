from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.quantile_forecast import (
    add_expanding_quantiles,
    apply_residual_quantiles,
    quantile_metrics,
    residual_quantile_offsets,
)


def test_quantile_metrics_reports_coverage_width_and_pinball() -> None:
    metrics = quantile_metrics(
        actuals=np.array([1.0, 2.0, 3.0, 4.0, 6.0]),
        q10=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        q50=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        q90=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    assert metrics["quantile_rows"] == 5
    assert metrics["interval_coverage_80"] == 1.0
    assert metrics["mean_interval_width_80"] == 2.0
    assert metrics["pinball_loss_q50"] == 0.1
    assert metrics["winkler_interval_score_80"] == 2.0


def test_residual_quantiles_are_ordered() -> None:
    offsets = residual_quantile_offsets([0, 10, 20], [10, 10, 10])
    quantiles = apply_residual_quantiles([5, 6], offsets)

    assert np.all(quantiles[:, 0] <= quantiles[:, 1])
    assert np.all(quantiles[:, 1] <= quantiles[:, 2])


def test_expanding_quantiles_use_only_prior_residuals() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "model": ["m"] * 4,
            "actual": [11.0, 12.0, 13.0, 50.0],
            "point": [10.0, 10.0, 10.0, 10.0],
        }
    )

    result = add_expanding_quantiles(
        frame,
        group_columns=["model"],
        actual_column="actual",
        point_column="point",
        output_prefix="prediction",
        min_calibration_rows=3,
    )

    assert result.loc[:2, "prediction_q10"].isna().all()
    assert result.loc[3, "prediction_q10"] == 11.2
    assert result.loc[3, "prediction_q50"] == 12.0
    assert result.loc[3, "prediction_q90"] == 12.8
