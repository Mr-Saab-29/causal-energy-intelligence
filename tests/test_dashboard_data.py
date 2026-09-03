from __future__ import annotations

import math

import pandas as pd

from scripts.build_dashboard_data import (
    build_active_future_causal_recommendations,
    build_active_future_recommendations,
    build_active_future_scenario_recommendations,
    enrich_scenario_recommendations,
    filter_future_recommendations,
    normalize_recommendation_fields,
    sanitize_json_value,
    summarize_marginal_shift_metrics,
)
from src.optimization.workload_shift import WorkloadConstraints, build_workload_decision_rankings


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


def test_normalize_recommendation_fields_defaults_missing_risk_status() -> None:
    frame = pd.DataFrame({"recommendation_rank": [1]})

    normalized = normalize_recommendation_fields(frame)

    assert normalized.loc[0, "recommendation_status"] == "recommended"
    assert not normalized.loc[0, "suppressed_by_uncertainty_guard"]
    assert "decision_uncertainty_score" in normalized


def test_normalize_recommendation_fields_renumbers_visible_scenario_ranks() -> None:
    frame = pd.DataFrame(
        {
            "scenario": ["balanced", "balanced", "clean_first"],
            "window": ["future_24h", "future_24h", "future_24h"],
            "model": ["model_a", "model_a", "model_a"],
            "decision_group": ["2026-08-25", "2026-08-25", "2026-08-25"],
            "timestamp_utc": [
                "2026-08-25T13:00:00+00:00",
                "2026-08-25T14:00:00+00:00",
                "2026-08-25T13:00:00+00:00",
            ],
            "recommendation_rank": [3, 4, 3],
        }
    )

    normalized = normalize_recommendation_fields(frame)
    balanced = normalized[normalized["scenario"] == "balanced"]

    assert balanced["recommendation_rank"].tolist() == [1, 2]
    assert normalized[normalized["scenario"] == "clean_first"]["recommendation_rank"].tolist() == [1]


def test_sanitize_json_value_replaces_non_finite_numbers() -> None:
    payload = {
        "missing": float("nan"),
        "infinite": float("inf"),
        "rows": [{"value": pd.NA}, {"value": pd.Timestamp("2026-08-25T01:00:00Z")}],
        "tuple": (1, math.nan),
    }

    sanitized = sanitize_json_value(payload)

    assert sanitized == {
        "missing": None,
        "infinite": None,
        "rows": [{"value": None}, {"value": "2026-08-25T01:00:00+00:00"}],
        "tuple": [1, None],
    }


def test_summarize_marginal_shift_metrics_prefers_future_report() -> None:
    report = {
        "historical": {
            "method": "marginal_proxy_mvp",
            "aggregate": {"top_1_change_share": 0.0},
        },
        "future": {
            "method": "marginal_proxy_mvp",
            "quality_guard": {"status": "warning", "warnings": ["low_marginal_proxy_coverage"]},
            "aggregate": {
                "top_1_change_share": 0.5,
                "mean_top_5_overlap_share": 0.8,
                "mean_absolute_rank_shift": 1.25,
                "mean_causal_adjustment_coverage": 0.75,
                "mean_top_1_regret_delta": -0.2,
            },
        },
    }

    summary = summarize_marginal_shift_metrics(report)

    assert summary == {
        "method": "marginal_proxy_mvp",
        "quality_status": "warning",
        "warnings": ["low_marginal_proxy_coverage"],
        "top_1_change_share": 0.5,
        "mean_top_5_overlap_share": 0.8,
        "mean_absolute_rank_shift": 1.25,
        "mean_causal_adjustment_coverage": 0.75,
        "mean_top_1_regret_delta": -0.2,
    }


def test_active_future_recommendations_refill_to_top5_after_past_rows_drop() -> None:
    rankings = sample_future_rankings()
    stale_top5 = rankings[rankings["predicted_decision_rank"] <= 5].copy()
    stale_top5["recommendation_rank"] = stale_top5["predicted_decision_rank"]

    active = build_active_future_recommendations(
        stale_top5,
        rankings,
        now=pd.Timestamp("2026-08-25T02:15:00Z"),
    )

    assert len(active) == 5
    assert active["recommendation_rank"].tolist() == [1, 2, 3, 4, 5]
    assert active["timestamp_utc"].min() >= pd.Timestamp("2026-08-25T02:00:00Z")


def test_active_future_recommendations_refill_handles_legacy_rankings() -> None:
    rankings = sample_future_rankings().drop(columns=["is_low_uncertainty_candidate"])
    stale_top5 = rankings[rankings["predicted_decision_rank"] <= 5].copy()
    stale_top5["recommendation_rank"] = stale_top5["predicted_decision_rank"]

    active = build_active_future_recommendations(
        stale_top5,
        rankings,
        now=pd.Timestamp("2026-08-25T02:15:00Z"),
    )

    assert len(active) == 5
    assert "recommendation_status" in active


def test_active_future_scenario_recommendations_refill_each_scenario_to_top5() -> None:
    rankings = sample_future_rankings()
    scenario_top5 = pd.DataFrame(
        {
            "scenario": ["balanced"] * 5,
            "window": ["future_24h"] * 5,
            "model": ["model_a"] * 5,
            "decision_group": ["2026-08-25"] * 5,
            "timestamp_utc": pd.date_range("2026-08-25", periods=5, freq="h", tz="UTC"),
            "recommendation_rank": [1, 2, 3, 4, 5],
        }
    )

    active = build_active_future_scenario_recommendations(
        scenario_top5,
        rankings,
        now=pd.Timestamp("2026-08-25T02:15:00Z"),
    )

    assert set(active["scenario"]) == {"balanced", "clean_first", "cost_aware_clean"}
    assert active.groupby("scenario").size().tolist() == [5, 5, 5]


def test_active_future_causal_recommendations_refill_preserves_proxy_context() -> None:
    rankings = sample_future_rankings()
    rankings["carbon_ranking_strategy"] = "marginal_proxy"
    rankings["causal_carbon_source"] = "marginal_emissions_proxy"
    rankings["causal_adjustment_available"] = True
    rankings["average_predicted_decision_rank"] = rankings["predicted_decision_rank"] + 1
    rankings["average_actual_decision_rank"] = rankings["actual_decision_rank"] + 1
    rankings["average_predicted_carbon_rank"] = rankings["predicted_carbon_rank"] + 1
    rankings["average_actual_carbon_rank"] = rankings["actual_carbon_rank"] + 1
    rankings["causal_adjusted_rank_shift"] = -1
    rankings["predicted_marginal_carbon_intensity_g_co2e_per_kwh"] = (
        rankings["predicted_avg_carbon_intensity_g_co2e_per_kwh"] + 10
    )
    rankings["actual_marginal_carbon_intensity_g_co2e_per_kwh"] = (
        rankings["actual_avg_carbon_intensity_g_co2e_per_kwh"] + 10
    )
    rankings["predicted_marginal_proxy_confidence"] = "medium"
    stale_top5 = rankings[rankings["predicted_decision_rank"] <= 5].copy()
    stale_top5["recommendation_rank"] = stale_top5["predicted_decision_rank"]

    active = build_active_future_causal_recommendations(
        stale_top5,
        rankings,
        now=pd.Timestamp("2026-08-25T02:15:00Z"),
    )

    assert len(active) == 5
    assert active["recommendation_rank"].tolist() == [1, 2, 3, 4, 5]
    assert active["carbon_ranking_strategy"].eq("marginal_proxy").all()
    assert "predicted_marginal_carbon_intensity_g_co2e_per_kwh" in active


def sample_future_rankings() -> pd.DataFrame:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-08-25", periods=10, freq="h", tz="UTC"),
            "window": ["future_24h"] * 10,
            "model": ["model_a"] * 10,
            "decision_date": ["2026-08-25"] * 10,
            "actual_price_eur_mwh": list(range(10)),
            "predicted_price_eur_mwh": list(range(10)),
            "previous_day_price_eur_mwh": [5] * 10,
            "actual_carbon_intensity_g_co2e_per_kwh": list(range(10)),
            "predicted_carbon_intensity_g_co2e_per_kwh": list(range(10)),
            "actual_total_emissions_kg_co2e": list(range(10)),
            "predicted_total_emissions_kg_co2e": list(range(10)),
        }
    )
    rankings = build_workload_decision_rankings(hourly, WorkloadConstraints())
    rankings["ranking_model_score"] = 1 - rankings["predicted_combined_score"]
    rankings["ranking_model_decision_score"] = rankings["predicted_combined_score"]
    rankings["ranking_model_score_source"] = "test"
    return rankings
