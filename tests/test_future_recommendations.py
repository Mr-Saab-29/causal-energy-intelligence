from __future__ import annotations

import pandas as pd

from src.models.future_recommendations import (
    append_operational_history,
    calculate_future_forecast_start,
    load_latest_operational_recommendation_snapshot,
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


def test_append_operational_history_rewrites_when_schema_changes(tmp_path) -> None:
    path = tmp_path / "operational_ranking_history.csv"
    first = pd.DataFrame(
        {
            "timestamp_utc": ["2026-08-10T08:00:00+00:00"],
            "model": ["model_a"],
            "old_column": [1],
        }
    )
    second = pd.DataFrame(
        {
            "timestamp_utc": ["2026-08-10T09:00:00+00:00"],
            "model": ["model_a"],
            "new_column": [2],
        }
    )

    append_operational_history(first, path, "2026-08-10T07:00:00+00:00")
    append_operational_history(second, path, "2026-08-10T08:00:00+00:00")

    history = pd.read_csv(path)
    assert history.columns.tolist() == [
        "timestamp_utc",
        "model",
        "old_column",
        "forecast_generated_at_utc",
        "new_column",
    ]
    assert len(history) == 2


def test_latest_operational_snapshot_ignores_malformed_history(tmp_path) -> None:
    path = tmp_path / "operational_recommendation_history.csv"
    path.write_text("a,b\n1,2,3\n", encoding="utf-8")

    snapshot = load_latest_operational_recommendation_snapshot(path)

    assert snapshot.empty
