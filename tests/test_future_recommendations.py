from __future__ import annotations

import pandas as pd

from src.models.future_recommendations import (
    calculate_future_forecast_start,
    remove_duplicate_columns,
)


def test_remove_duplicate_columns_preserves_first_occurrence() -> None:
    frame = pd.DataFrame(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        columns=["timestamp_utc", "wind_lag_24h", "wind_lag_24h"],
    )

    result = remove_duplicate_columns(frame)

    assert result.columns.tolist() == ["timestamp_utc", "wind_lag_24h"]
    assert result["wind_lag_24h"].tolist() == [2, 5]


def test_calculate_future_forecast_start_uses_current_future_hour_when_data_lags() -> None:
    result = calculate_future_forecast_start(
        pd.Timestamp("2026-08-09T16:00:00Z"),
        as_of_utc="2026-08-10T08:27:00Z",
    )

    assert result == pd.Timestamp("2026-08-10T09:00:00Z")
