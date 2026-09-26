from __future__ import annotations

import pandas as pd

from src.data.causal_data_quality import evaluate_causal_data_metrics
from src.data.entsoe_causal_ingest import REQUIRED_CAUSAL_GRID_TABLES
from src.data.source_config import FRANCE_DIRECT_ENTSOE_NEIGHBORS


def complete_metrics() -> dict[str, object]:
    cross_border = []
    for metric in ("physical_flow", "scheduled_exchange"):
        for neighbor in FRANCE_DIRECT_ENTSOE_NEIGHBORS:
            for source, target in (("FR", neighbor), (neighbor, "FR")):
                cross_border.append(
                    {
                        "metric": metric,
                        "from_bidding_zone": source,
                        "to_bidding_zone": target,
                        "granularity": "15m",
                        "row_count": 96,
                        "timestamp_count": 96,
                        "hourly_timestamp_count": 24,
                    }
                )
    cross_border.append(
        {
            "metric": "day_ahead_capacity",
            "from_bidding_zone": "FR",
            "to_bidding_zone": "ES",
            "granularity": "1h",
            "row_count": 24,
            "timestamp_count": 24,
            "hourly_timestamp_count": 24,
        }
    )
    return {
        "cross_border_series": cross_border,
        "forecast_series": [
            {
                "forecast_type": forecast_type,
                "granularity": "15m",
                "row_count": 96,
                "timestamp_count": 96,
                "hourly_timestamp_count": 24,
                "negative_value_count": 0,
            }
            for forecast_type in ("load", "wind_onshore", "wind_offshore", "solar")
        ],
        "outage_summary": [
            {
                "outage_type": "planned",
                "row_count": 3,
                "outage_count": 3,
                "missing_unavailable_capacity_count": 0,
            }
        ],
        "balancing_series": [
            {
                "metric": metric,
                "granularity": "15m",
                "row_count": 96,
                "timestamp_count": 96,
                "hourly_timestamp_count": 24,
            }
            for metric in (
                "imbalance_price",
                "imbalance_volume",
                "activated_energy_price",
            )
        ],
        "duplicate_source_key_groups": {
            table: 0 for table in REQUIRED_CAUSAL_GRID_TABLES
        },
        "temporal_violations": {
            "operational_forecast_after_target": 0,
            "operational_schedule_after_target": 0,
            "operational_outage_published_after_snapshot": 0,
            "historical_forecast_claims_generation_time": 0,
            "historical_cross_border_has_snapshot_suffix": 0,
        },
        "storage_bytes": {},
        "total_storage_bytes": 0,
    }


def test_complete_historical_window_is_backfill_ready() -> None:
    evaluation = evaluate_causal_data_metrics(
        complete_metrics(),
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-02", tz="UTC"),
        vintage_quality="historical_final",
        min_coverage=0.95,
    )

    assert evaluation["status"] == "pass"
    assert evaluation["backfill_ready"] is True
    assert evaluation["critical_issues"] == []
    assert "activated_energy_unavailable_from_legacy_a83_endpoint" in evaluation[
        "known_limitations"
    ]
    assert any(
        limitation.startswith("day_ahead_capacity_not_published_for_pairs:")
        for limitation in evaluation["known_limitations"]
    )


def test_missing_required_series_and_unclassified_outage_block_backfill() -> None:
    metrics = complete_metrics()
    metrics["forecast_series"] = [
        row for row in metrics["forecast_series"] if row["forecast_type"] != "solar"
    ]
    metrics["outage_summary"] = [
        {
            "outage_type": "other",
            "row_count": 4,
            "outage_count": 4,
            "missing_unavailable_capacity_count": 0,
        }
    ]
    metrics["temporal_violations"]["historical_forecast_claims_generation_time"] = 2

    evaluation = evaluate_causal_data_metrics(
        metrics,
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-02", tz="UTC"),
        vintage_quality="historical_final",
        min_coverage=0.95,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["backfill_ready"] is False
    assert "missing_forecast_type:solar" in evaluation["critical_issues"]
    assert "unclassified_outage_rows:4" in evaluation["critical_issues"]
    assert (
        "temporal_violation:historical_forecast_claims_generation_time:2"
        in evaluation["critical_issues"]
    )


def test_low_resolution_aware_coverage_blocks_backfill() -> None:
    metrics = complete_metrics()
    metrics["forecast_series"][0]["hourly_timestamp_count"] = 20

    evaluation = evaluate_causal_data_metrics(
        metrics,
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-02", tz="UTC"),
        vintage_quality="historical_final",
        min_coverage=0.95,
    )

    assert evaluation["backfill_ready"] is False
    assert "low_forecast_coverage:load:0.8333" in evaluation["critical_issues"]


def test_projected_storage_over_budget_blocks_backfill() -> None:
    metrics = complete_metrics()
    metrics["storage_projection"] = {"total_projected_storage_bytes": 2_000}

    evaluation = evaluate_causal_data_metrics(
        metrics,
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-02", tz="UTC"),
        vintage_quality="historical_final",
        min_coverage=0.95,
        max_projected_storage_bytes=1_000,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["backfill_ready"] is False
    assert "projected_storage_exceeds_budget:2000>1000" in evaluation[
        "critical_issues"
    ]


def test_mixed_native_resolution_passes_when_every_decision_hour_exists() -> None:
    metrics = complete_metrics()
    target = next(
        row
        for row in metrics["cross_border_series"]
        if row["metric"] == "physical_flow"
        and row["from_bidding_zone"] == "FR"
        and row["to_bidding_zone"] == "DE_LU"
    )
    target["row_count"] = 55
    target["timestamp_count"] = 55
    target["hourly_timestamp_count"] = 24

    evaluation = evaluate_causal_data_metrics(
        metrics,
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-02", tz="UTC"),
        vintage_quality="historical_final",
        min_coverage=0.95,
    )

    assert evaluation["backfill_ready"] is True
    assert evaluation["coverage"]["cross_border"]["physical_flow|FR|DE_LU"] == 1.0
