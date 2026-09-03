from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

from src.causal.recommendations import (
    build_causal_adjusted_recommendations,
    build_marginal_workload_rankings,
    run_all_causal_adjusted_recommendations,
    run_causal_adjusted_recommendations,
    summarize_ranking_shifts,
)
import src.causal.recommendations as causal_recommendations
from src.optimization.workload_shift import WorkloadConstraints


def test_marginal_rankings_keep_average_rank_context() -> None:
    rankings = sample_rankings()
    marginal = build_marginal_workload_rankings(
        rankings,
        WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    second = marginal[marginal["timestamp_utc"] == pd.Timestamp("2026-01-01T01:00:00Z")].iloc[0]
    third = marginal[marginal["timestamp_utc"] == pd.Timestamp("2026-01-01T02:00:00Z")].iloc[0]

    assert second["average_predicted_decision_rank"] == 2
    assert second["predicted_decision_rank"] == 1
    assert third["average_predicted_decision_rank"] == 1
    assert third["predicted_decision_rank"] == 2
    assert second["causal_carbon_source"] == "marginal_emissions_proxy"


def test_shift_metrics_include_quality_guard_warning_for_low_coverage() -> None:
    rankings = sample_rankings()
    marginal = build_marginal_workload_rankings(
        rankings,
        WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    metrics = summarize_ranking_shifts(rankings, marginal, top_n=2)

    assert metrics["aggregate"]["groups"] == 1
    assert metrics["aggregate"]["top_1_change_share"] == 1.0
    assert metrics["quality_guard"]["status"] == "warning"
    assert metrics["quality_guard"]["warnings"] == ["low_marginal_proxy_coverage"]


def test_causal_adjusted_recommendations_export_mvp_context() -> None:
    marginal = build_marginal_workload_rankings(
        sample_rankings(),
        WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    recommendations = build_causal_adjusted_recommendations(marginal, top_n=2)

    assert recommendations["timestamp_utc"].tolist() == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T02:00:00Z"),
    ]
    assert recommendations.loc[0, "carbon_ranking_strategy"] == "marginal_proxy"
    assert recommendations.loc[0, "causal_adjusted_rank_shift"] == -1


def test_run_causal_adjusted_recommendations_writes_outputs(tmp_path) -> None:
    rankings_path = tmp_path / "rankings.csv"
    output_rankings_path = tmp_path / "marginal.csv"
    output_recommendations_path = tmp_path / "recommendations.csv"
    output_metrics_path = tmp_path / "metrics.json"
    sample_rankings().to_csv(rankings_path, index=False)

    result = run_causal_adjusted_recommendations(
        rankings_path,
        output_rankings_path,
        output_recommendations_path,
        output_metrics_path,
        top_n=2,
    )

    assert result["aggregate"]["groups"] == 1
    assert len(pd.read_csv(output_rankings_path)) == 3
    assert len(pd.read_csv(output_recommendations_path)) == 2
    json.loads(output_metrics_path.read_text(encoding="utf-8"))


def test_run_all_causal_adjusted_recommendations_skips_missing_historical(tmp_path) -> None:
    future_rankings_path = tmp_path / "future_rankings.csv"
    future_ranking_output_path = tmp_path / "future_marginal.csv"
    future_recommendation_output_path = tmp_path / "future_recommendations.csv"
    metrics_output_path = tmp_path / "metrics.json"
    sample_rankings().to_csv(future_rankings_path, index=False)

    with (
        patch.object(
            causal_recommendations,
            "DEFAULT_HISTORICAL_RANKINGS_PATH",
            tmp_path / "missing_historical.csv",
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_HISTORICAL_RANKING_OUTPUT_PATH",
            tmp_path / "historical_marginal.csv",
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_HISTORICAL_RECOMMENDATION_OUTPUT_PATH",
            tmp_path / "historical_recommendations.csv",
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RANKINGS_PATH",
            future_rankings_path,
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RANKING_OUTPUT_PATH",
            future_ranking_output_path,
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH",
            future_recommendation_output_path,
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_METRICS_OUTPUT_PATH",
            metrics_output_path,
        ),
    ):
        result = run_all_causal_adjusted_recommendations(top_n=2)

    assert result["historical"]["status"] == "skipped"
    assert future_ranking_output_path.exists()
    assert future_recommendation_output_path.exists()
    assert metrics_output_path.exists()


def sample_rankings() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "workload_end_utc": timestamps + pd.Timedelta(hours=1),
            "window": ["test"] * 3,
            "model": ["model_a"] * 3,
            "decision_date": ["2026-01-01"] * 3,
            "decision_group": ["2026-01-01"] * 3,
            "duration_hours": [1] * 3,
            "actual_avg_price_eur_mwh": [1.0, 2.0, 3.0],
            "predicted_avg_price_eur_mwh": [1.0, 2.0, 3.0],
            "previous_day_avg_price_eur_mwh": [1.0, 1.0, 1.0],
            "actual_avg_carbon_intensity_g_co2e_per_kwh": [30.0, 20.0, 10.0],
            "predicted_avg_carbon_intensity_g_co2e_per_kwh": [30.0, 20.0, 10.0],
            "actual_total_emissions_kg_co2e": [300.0, 400.0, 300.0],
            "predicted_total_emissions_kg_co2e": [300.0, 400.0, 300.0],
            "predicted_price_rank": [1, 2, 3],
            "predicted_carbon_rank": [3, 2, 1],
            "actual_price_rank": [1, 2, 3],
            "actual_carbon_rank": [3, 2, 1],
            "candidate_count": [3, 3, 3],
            "predicted_price_rank_pct": [0.0, 0.5, 1.0],
            "predicted_carbon_rank_pct": [1.0, 0.5, 0.0],
            "actual_price_rank_pct": [0.0, 0.5, 1.0],
            "actual_carbon_rank_pct": [1.0, 0.5, 0.0],
            "predicted_combined_score": [1.0, 0.5, 0.0],
            "actual_combined_score": [1.0, 0.5, 0.0],
            "predicted_decision_rank": [3, 2, 1],
            "actual_decision_rank": [3, 2, 1],
            "combined_regret": [1.0, 0.5, 0.0],
            "cost_regret_eur_mwh": [0.0, 1.0, 2.0],
            "carbon_regret_g_co2e_per_kwh": [0.0, 10.0, 20.0],
            "cost_savings_vs_run_now_eur_mwh": [0.0, -1.0, -2.0],
            "carbon_savings_vs_run_now_g_co2e_per_kwh": [0.0, -10.0, -20.0],
            "baseline_predicted_decision_rank": [3, 2, 1],
            "ranking_model_score": [0.0, 0.5, 1.0],
            "ranking_model_decision_score": [1.0, 0.5, 0.0],
            "ranking_model_score_source": ["test"] * 3,
            "decision_uncertainty_score": [0.0, 0.0, 0.0],
            "prediction_interval_uncertainty_score": [0.0, 0.0, 0.0],
            "uncertainty_guard_penalty": [0.0, 0.0, 0.0],
            "uncertainty_guard_applied": [False, False, False],
            "is_low_uncertainty_candidate": [True, True, True],
            "predicted_price_direction_vs_previous_day": ["flat", "increase", "increase"],
        }
    )
