from __future__ import annotations

import pandas as pd

from scripts.build_dashboard_data import (
    enrich_scenario_recommendations,
    filter_future_recommendations,
)


def test_filter_future_recommendations_drops_past_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-08-10T07:00:00+00:00",
                "2026-08-10T08:00:00+00:00",
                "2026-08-10T09:00:00+00:00",
            ],
            "recommendation_rank": [1, 2, 3],
        }
    )

    filtered = filter_future_recommendations(
        frame,
        now=pd.Timestamp("2026-08-10T08:10:00Z"),
    )

    assert filtered["timestamp_utc"].tolist() == [
        "2026-08-10T08:00:00+00:00",
        "2026-08-10T09:00:00+00:00",
    ]


def test_enrich_scenario_recommendations_adds_confidence_context() -> None:
    scenario_frame = pd.DataFrame(
        {
            "decision_group": ["2026-08-10"],
            "timestamp_utc": ["2026-08-10T08:00:00+00:00"],
            "scenario": ["clean_first"],
            "recommendation_rank": [1],
        }
    )
    recommendation_frame = pd.DataFrame(
        {
            "decision_group": ["2026-08-10"],
            "timestamp_utc": ["2026-08-10T08:00:00+00:00"],
            "confidence_score": [0.84],
            "confidence_level": ["high"],
            "candidate_count": [24],
        }
    )

    enriched = enrich_scenario_recommendations(
        scenario_frame,
        recommendation_frame,
    )

    assert enriched.loc[0, "confidence_score"] == 0.84
    assert enriched.loc[0, "confidence_level"] == "high"
    assert enriched.loc[0, "candidate_count"] == 24


def test_enrich_scenario_recommendations_keeps_existing_scenario_confidence() -> None:
    scenario_frame = pd.DataFrame(
        {
            "decision_group": ["2026-08-10"],
            "timestamp_utc": ["2026-08-10T08:00:00+00:00"],
            "scenario": ["clean_first"],
            "confidence_score": [0.62],
            "confidence_level": ["medium"],
        }
    )
    recommendation_frame = pd.DataFrame(
        {
            "decision_group": ["2026-08-10"],
            "timestamp_utc": ["2026-08-10T08:00:00+00:00"],
            "confidence_score": [0.84],
            "confidence_level": ["high"],
            "candidate_count": [24],
        }
    )

    enriched = enrich_scenario_recommendations(
        scenario_frame,
        recommendation_frame,
    )

    assert enriched.loc[0, "confidence_score"] == 0.62
    assert enriched.loc[0, "confidence_level"] == "medium"
    assert enriched.loc[0, "candidate_count"] == 24
    assert "confidence_score_x" not in enriched
