from __future__ import annotations

import pandas as pd

from src.models.future_recommendations import (
    append_operational_history,
    build_future_hourly_decision_inputs,
    calculate_future_forecast_start,
    load_latest_operational_recommendation_snapshot,
    prune_operational_history,
    remove_duplicate_columns,
)
from src.models.baseline_price import PRODUCTION_SIGNAL_TARGETS


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


def test_future_hourly_inputs_emit_source_generation_columns() -> None:
    timestamps = pd.date_range("2026-08-10T00:00:00Z", periods=2, freq="h")
    history = pd.DataFrame(
        {
            "timestamp_utc": timestamps - pd.Timedelta(days=1),
            "price_eur_mwh": [40.0, 42.0],
        }
    )
    future = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "model": ["model_a", "model_a"],
            "predicted_price_eur_mwh": [50.0, 55.0],
        }
    )
    for index, source in enumerate(PRODUCTION_SIGNAL_TARGETS[1:], start=1):
        future[f"forecast_{source}_mwh"] = [100.0 + index, 110.0 + index]

    hourly = build_future_hourly_decision_inputs(history, future)

    assert "actual_gas_generation_mwh" in hourly
    assert "predicted_gas_generation_mwh" in hourly
    assert hourly["actual_gas_generation_mwh"].tolist() == [102.0, 112.0]
    assert hourly["predicted_gas_generation_mwh"].tolist() == [102.0, 112.0]


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


def test_prune_operational_history_keeps_recent_rows() -> None:
    history = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-08-01T00:00:00Z",
                    "2026-09-01T00:00:00Z",
                    "2026-09-09T00:00:00Z",
                ],
                utc=True,
            ),
            "model": ["model_a", "model_a", "model_a"],
        }
    )

    pruned = prune_operational_history(history, retention_days=10)

    assert pruned["timestamp_utc"].astype(str).tolist() == [
        "2026-09-01 00:00:00+00:00",
        "2026-09-09 00:00:00+00:00",
    ]
