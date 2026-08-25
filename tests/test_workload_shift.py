from __future__ import annotations

import json

import pandas as pd
import pytest

from src.optimization.workload_shift import (
    WorkloadConstraints,
    apply_prediction_interval_uncertainty,
    build_confidence_calibration,
    build_prediction_interval_calibration,
    build_top_workload_recommendations,
    build_scenario_rerankings,
    build_workload_decision_rankings,
    select_champion_model,
    select_scenario_champions,
    summarize_policy_backtest,
    summarize_recommendation_drift,
    validate_columns,
)


def test_workload_rankings_penalize_high_uncertainty_candidates() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-08-10", periods=6, freq="h", tz="UTC"),
            "window": ["test"] * 6,
            "model": ["model_a"] * 6,
            "decision_date": ["2026-08-10"] * 6,
            "actual_price_eur_mwh": [1, 2, 3, 4, 5, 6],
            "predicted_price_eur_mwh": [1.00, 1.01, 3.0, 4.0, 5.0, 6.0],
            "previous_day_price_eur_mwh": [2, 2, 2, 2, 2, 2],
            "actual_carbon_intensity_g_co2e_per_kwh": [1, 2, 3, 4, 5, 6],
            "predicted_carbon_intensity_g_co2e_per_kwh": [1, 2, 3, 4, 5, 6],
            "actual_total_emissions_kg_co2e": [1, 2, 3, 4, 5, 6],
            "predicted_total_emissions_kg_co2e": [1, 2, 3, 4, 5, 6],
        }
    )

    rankings = build_workload_decision_rankings(
        hourly,
        WorkloadConstraints(price_weight=1.0, carbon_weight=0.0),
    )

    close_call = rankings[rankings["timestamp_utc"] == pd.Timestamp("2026-08-10T00:00:00Z")].iloc[0]
    separated = rankings[rankings["timestamp_utc"] == pd.Timestamp("2026-08-10T02:00:00Z")].iloc[0]

    assert close_call["decision_uncertainty_score"] > separated["decision_uncertainty_score"]
    assert close_call["uncertainty_guard_applied"]


def test_scenario_reranking_exports_top5_metrics_and_confidence() -> None:
    rankings = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-08-10", periods=6, freq="h", tz="UTC"),
            "workload_end_utc": pd.date_range("2026-08-10T01:00:00Z", periods=6, freq="h"),
            "window": ["test"] * 6,
            "model": ["model_a"] * 6,
            "decision_group": ["2026-08-10"] * 6,
            "duration_hours": [1] * 6,
            "predicted_price_rank_pct": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "predicted_carbon_rank_pct": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "actual_price_rank_pct": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "actual_carbon_rank_pct": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "predicted_price_direction_vs_previous_day": ["increase"] * 6,
            "predicted_avg_carbon_intensity_g_co2e_per_kwh": [1, 2, 3, 4, 5, 6],
            "predicted_total_emissions_kg_co2e": [1, 2, 3, 4, 5, 6],
            "predicted_carbon_rank": [1, 2, 3, 4, 5, 6],
            "carbon_regret_g_co2e_per_kwh": [0, 1, 2, 3, 4, 5],
            "cost_regret_eur_mwh": [0, 1, 2, 3, 4, 5],
            "carbon_savings_vs_run_now_g_co2e_per_kwh": [0, -1, -2, -3, -4, -5],
        }
    )

    recommendations, metrics = build_scenario_rerankings(rankings, top_n=5)

    assert "confidence_score" in recommendations
    assert "decision_uncertainty_score" in recommendations
    assert metrics["summary"][0]["top_5_f1"] == 1.0
    assert metrics["summary"][0]["pairwise_ranking_loss"] == 0.0


def test_confidence_calibration_can_be_grouped_by_scenario() -> None:
    recommendations = pd.DataFrame(
        {
            "scenario": ["clean_first", "cost_aware_clean"],
            "confidence_score": [0.8, 0.8],
            "actual_decision_rank": [1, 6],
            "combined_regret": [0.0, 0.4],
            "carbon_regret_g_co2e_per_kwh": [0.0, 2.0],
            "cost_regret_eur_mwh": [0.0, 3.0],
        }
    )

    calibration = build_confidence_calibration(
        recommendations,
        top_n=5,
        group_column="scenario",
        min_bin_rows=1,
        min_group_rows=1,
    )

    assert calibration["groups"]["clean_first"][0]["empirical_top_n_hit_rate"] == 1.0
    assert calibration["groups"]["cost_aware_clean"][0]["empirical_top_n_hit_rate"] == 0.0


def test_champion_selection_prioritizes_recommendation_regret_over_mae(tmp_path) -> None:
    price_metrics_path = tmp_path / "price.json"
    carbon_metrics_path = tmp_path / "carbon.json"
    price_metrics_path.write_text(
        json.dumps({"summary": [{"model": "low_mae"}, {"model": "low_regret"}]}),
        encoding="utf-8",
    )
    carbon_metrics_path.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "model": "low_mae",
                        "methodology": "direct_operational_emissions",
                        "carbon_intensity_mae_g_co2e_per_kwh": 0.1,
                    },
                    {
                        "model": "low_regret",
                        "methodology": "direct_operational_emissions",
                        "carbon_intensity_mae_g_co2e_per_kwh": 10.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ranking_metrics = [
        {
            "model": "low_mae",
            "price_direction_error": 0.1,
            "mean_top_1_carbon_regret_g_co2e_per_kwh": 0.1,
            "pairwise_ranking_loss": 0.1,
            "top_5_f1": 0.9,
            "mean_top_1_combined_regret": 1.0,
        },
        {
            "model": "low_regret",
            "price_direction_error": 0.1,
            "mean_top_1_carbon_regret_g_co2e_per_kwh": 0.2,
            "pairwise_ranking_loss": 0.1,
            "top_5_f1": 0.9,
            "mean_top_1_combined_regret": 0.0,
        },
    ]

    champion = select_champion_model(
        price_metrics_path,
        carbon_metrics_path,
        ranking_metrics,
        methodology="direct_operational_emissions",
    )

    assert champion["champion_model"] == "low_regret"


def test_top_recommendations_emit_no_low_risk_status_when_all_candidates_uncertain() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-08-10", periods=3, freq="h", tz="UTC"),
            "window": ["test"] * 3,
            "model": ["model_a"] * 3,
            "decision_date": ["2026-08-10"] * 3,
            "actual_price_eur_mwh": [1, 2, 3],
            "predicted_price_eur_mwh": [1, 1, 1],
            "previous_day_price_eur_mwh": [2, 2, 2],
            "actual_carbon_intensity_g_co2e_per_kwh": [1, 2, 3],
            "predicted_carbon_intensity_g_co2e_per_kwh": [1, 1, 1],
            "actual_total_emissions_kg_co2e": [1, 2, 3],
            "predicted_total_emissions_kg_co2e": [1, 1, 1],
        }
    )
    rankings = build_workload_decision_rankings(hourly, WorkloadConstraints())

    recommendations = build_top_workload_recommendations(rankings, top_n=5)

    assert recommendations["recommendation_status"].tolist() == [
        "no_low_risk_recommendation_available"
    ]
    assert recommendations["suppressed_by_uncertainty_guard"].tolist() == [True]


def test_default_confidence_calibration_skips_small_scenario_bins() -> None:
    recommendations = pd.DataFrame(
        {
            "scenario": ["clean_first"],
            "confidence_score": [0.8],
            "actual_decision_rank": [1],
            "combined_regret": [0.0],
            "carbon_regret_g_co2e_per_kwh": [0.0],
            "cost_regret_eur_mwh": [0.0],
        }
    )

    calibration = build_confidence_calibration(
        recommendations,
        top_n=5,
        group_column="scenario",
    )

    assert calibration["groups"] == {}
    assert calibration["bins"] == []


def test_validate_columns_raises_clear_artifact_error() -> None:
    with pytest.raises(ValueError, match="test artifact missing required columns"):
        validate_columns(pd.DataFrame({"present": [1]}), ["present", "missing"], "test artifact")


def test_recommendation_drift_reports_uncertainty_and_status_counts() -> None:
    recommendations = pd.DataFrame(
        {
            "decision_group": ["2026-08-10", "2026-08-10"],
            "timestamp_utc": [
                "2026-08-10T08:00:00+00:00",
                "2026-08-10T09:00:00+00:00",
            ],
            "confidence_score": [0.9, 0.4],
            "confidence_level": ["high", "low"],
            "decision_uncertainty_score": [0.2, 0.9],
            "predicted_avg_carbon_intensity_g_co2e_per_kwh": [10.0, 20.0],
            "recommendation_status": [
                "recommended",
                "no_low_risk_recommendation_available",
            ],
        }
    )

    drift = summarize_recommendation_drift(recommendations)

    assert drift["recommendations"]["high_confidence_share"] == 0.5
    assert drift["recommendations"]["high_uncertainty_share"] == 0.5
    assert drift["recommendations"]["recommendation_status_counts"] == {
        "recommended": 1,
        "no_low_risk_recommendation_available": 1,
    }


def test_prediction_interval_calibration_adds_interval_uncertainty() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-08-10", periods=6, freq="h", tz="UTC"),
            "window": ["test"] * 6,
            "model": ["model_a"] * 6,
            "decision_date": ["2026-08-10"] * 6,
            "actual_price_eur_mwh": [10, 11, 12, 13, 14, 15],
            "predicted_price_eur_mwh": [12, 13, 14, 15, 16, 17],
            "previous_day_price_eur_mwh": [10, 10, 10, 10, 10, 10],
            "actual_carbon_intensity_g_co2e_per_kwh": [1, 2, 3, 4, 5, 6],
            "predicted_carbon_intensity_g_co2e_per_kwh": [2, 3, 4, 5, 6, 7],
            "actual_total_emissions_kg_co2e": [1, 2, 3, 4, 5, 6],
            "predicted_total_emissions_kg_co2e": [2, 3, 4, 5, 6, 7],
        }
    )
    rankings = build_workload_decision_rankings(hourly, WorkloadConstraints())
    calibration = build_prediction_interval_calibration(rankings, quantile=0.5)

    interval_rankings = apply_prediction_interval_uncertainty(rankings, calibration)

    assert calibration["models"]["model_a"]["price_interval_half_width_eur_mwh"] > 0
    assert "prediction_interval_uncertainty_score" in interval_rankings
    assert interval_rankings["predicted_price_interval_half_width_eur_mwh"].gt(0).all()


def test_policy_backtest_summarizes_base_and_scenario_recommendations() -> None:
    base = pd.DataFrame(
        {
            "model": ["model_a"],
            "decision_group": ["2026-08-10"],
            "recommendation_rank": [1],
            "recommendation_status": ["recommended"],
            "actual_decision_rank": [2],
            "combined_regret": [0.2],
            "carbon_regret_g_co2e_per_kwh": [1.0],
            "cost_regret_eur_mwh": [3.0],
            "confidence_score": [0.8],
            "decision_uncertainty_score": [0.3],
        }
    )
    scenario = base.assign(scenario="clean_first", actual_scenario_rank=1)

    backtest = summarize_policy_backtest(base, scenario)

    assert backtest["base_policy"][0]["top_5_hit_rate"] == 1.0
    assert backtest["scenario_policy"][0]["scenario"] == "clean_first"


def test_select_scenario_champions_selects_lowest_scenario_score() -> None:
    champions = select_scenario_champions(
        [
            {
                "scenario": "clean_first",
                "model": "high_regret",
                "mean_scenario_regret": 1.0,
                "mean_carbon_regret_g_co2e_per_kwh": 1.0,
                "top_5_f1": 0.5,
            },
            {
                "scenario": "clean_first",
                "model": "low_regret",
                "mean_scenario_regret": 0.0,
                "mean_carbon_regret_g_co2e_per_kwh": 0.0,
                "top_5_f1": 1.0,
            },
        ]
    )

    assert champions["champions"][0]["model"] == "low_regret"
