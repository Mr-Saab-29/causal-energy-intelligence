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
from src.optimization.workload_shift import WorkloadConstraints, build_workload_decision_rankings


def test_marginal_rankings_replace_carbon_order_without_mutating_average_rank() -> None:
    average_rankings = sample_average_rankings()
    marginal = build_marginal_workload_rankings(
        average_rankings,
        sample_marginal_proxy(),
        constraints=WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    first = marginal[marginal["timestamp_utc"] == pd.Timestamp("2026-01-01T00:00:00Z")].iloc[0]
    third = marginal[marginal["timestamp_utc"] == pd.Timestamp("2026-01-01T02:00:00Z")].iloc[0]

    assert first["average_predicted_decision_rank"] == 1
    assert first["predicted_decision_rank"] == 3
    assert third["average_predicted_decision_rank"] == 3
    assert third["predicted_decision_rank"] == 1
    assert third["causal_carbon_source"] == "marginal_emissions_proxy"


def test_shift_metrics_quantify_top_changes() -> None:
    average_rankings = sample_average_rankings()
    marginal_rankings = build_marginal_workload_rankings(
        average_rankings,
        sample_marginal_proxy(),
        constraints=WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    metrics = summarize_ranking_shifts(average_rankings, marginal_rankings, top_n=2)

    assert metrics["aggregate"]["groups"] == 1
    assert metrics["aggregate"]["top_1_change_share"] == 1.0
    assert metrics["aggregate"]["mean_top_5_overlap_share"] == 1 / 3
    assert metrics["method"] == "marginal_proxy_mvp"
    assert metrics["quality_guard"]["status"] == "ok"
    assert metrics["summary"][0]["top_1_average_timestamp_utc"] == "2026-01-01T00:00:00+00:00"
    assert metrics["summary"][0]["top_1_marginal_timestamp_utc"] == "2026-01-01T02:00:00+00:00"


def test_causal_adjusted_recommendations_use_marginal_rank_order() -> None:
    average_rankings = sample_average_rankings()
    marginal_rankings = build_marginal_workload_rankings(
        average_rankings,
        sample_marginal_proxy(),
        constraints=WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    recommendations = build_causal_adjusted_recommendations(marginal_rankings, top_n=2)

    assert recommendations["timestamp_utc"].tolist() == [
        pd.Timestamp("2026-01-01T02:00:00Z"),
        pd.Timestamp("2026-01-01T01:00:00Z"),
    ]
    assert recommendations.loc[0, "carbon_ranking_strategy"] == "marginal_proxy"
    assert recommendations.loc[0, "predicted_marginal_carbon_intensity_g_co2e_per_kwh"] == 10.0
    assert recommendations.loc[0, "causal_adjusted_rank_shift"] == -2


def test_run_causal_adjusted_recommendations_writes_outputs(tmp_path) -> None:
    average_path = tmp_path / "average.csv"
    marginal_path = tmp_path / "marginal.csv"
    ranking_path = tmp_path / "ranking.csv"
    recommendation_path = tmp_path / "recommendations.csv"
    metrics_path = tmp_path / "metrics.json"
    sample_average_rankings().to_csv(average_path, index=False)
    sample_marginal_proxy().to_csv(marginal_path, index=False)

    result = run_causal_adjusted_recommendations(
        average_rankings_path=average_path,
        marginal_proxy_path=marginal_path,
        ranking_output_path=ranking_path,
        recommendation_output_path=recommendation_path,
        metrics_output_path=metrics_path,
        top_n=2,
        ensure_marginal_proxy=False,
    )

    assert result["aggregate"]["groups"] == 1
    assert len(pd.read_csv(ranking_path)) == 3
    assert len(pd.read_csv(recommendation_path)) == 2
    json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "NaN" not in metrics_path.read_text(encoding="utf-8")


def test_rankings_from_average_proxy_warn_when_coverage_is_low() -> None:
    average_rankings = sample_average_rankings()
    marginal_rankings = build_marginal_workload_rankings(
        average_rankings,
        pd.DataFrame(),
        constraints=WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )

    metrics = summarize_ranking_shifts(average_rankings, marginal_rankings, top_n=2)

    assert metrics["quality_guard"]["status"] == "warning"
    assert metrics["quality_guard"]["warnings"] == ["low_marginal_proxy_coverage"]
    assert "predicted_marginal_carbon_intensity_g_co2e_per_kwh" in marginal_rankings


def test_run_all_causal_adjusted_recommendations_writes_future_outputs(tmp_path) -> None:
    historical_path = tmp_path / "historical.csv"
    future_path = tmp_path / "future.csv"
    proxy_path = tmp_path / "proxy.csv"
    future_ranking_path = tmp_path / "future_marginal.csv"
    future_recommendation_path = tmp_path / "future_recommendations.csv"
    metrics_path = tmp_path / "metrics.json"
    sample_average_rankings().to_csv(historical_path, index=False)
    sample_average_rankings().to_csv(future_path, index=False)
    sample_marginal_proxy().to_csv(proxy_path, index=False)

    with (
        patch.object(causal_recommendations, "DEFAULT_AVERAGE_RANKINGS_PATH", str(historical_path)),
        patch.object(causal_recommendations, "DEFAULT_FUTURE_AVERAGE_RANKINGS_PATH", str(future_path)),
        patch.object(causal_recommendations, "DEFAULT_MARGINAL_PROXY_PATH", str(proxy_path)),
        patch.object(
            causal_recommendations,
            "DEFAULT_RANKING_OUTPUT_PATH",
            str(tmp_path / "historical_marginal.csv"),
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_RECOMMENDATION_OUTPUT_PATH",
            str(tmp_path / "historical_recommendations.csv"),
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RANKING_OUTPUT_PATH",
            str(future_ranking_path),
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH",
            str(future_recommendation_path),
        ),
        patch.object(causal_recommendations, "DEFAULT_METRICS_OUTPUT_PATH", str(metrics_path)),
    ):
        result = run_all_causal_adjusted_recommendations(top_n=2)

    assert result["future"]["aggregate"]["groups"] == 1
    assert future_ranking_path.exists()
    assert future_recommendation_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["quality_guard"]["status"]


def test_run_all_causal_adjusted_recommendations_skips_missing_historical(tmp_path) -> None:
    future_path = tmp_path / "future.csv"
    future_ranking_path = tmp_path / "future_marginal.csv"
    future_recommendation_path = tmp_path / "future_recommendations.csv"
    metrics_path = tmp_path / "metrics.json"
    sample_average_rankings().to_csv(future_path, index=False)

    with (
        patch.object(
            causal_recommendations,
            "DEFAULT_AVERAGE_RANKINGS_PATH",
            str(tmp_path / "missing_historical.csv"),
        ),
        patch.object(causal_recommendations, "DEFAULT_FUTURE_AVERAGE_RANKINGS_PATH", str(future_path)),
        patch.object(causal_recommendations, "DEFAULT_MARGINAL_PROXY_PATH", str(tmp_path / "proxy.csv")),
        patch.object(
            causal_recommendations,
            "DEFAULT_RANKING_OUTPUT_PATH",
            str(tmp_path / "historical_marginal.csv"),
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_RECOMMENDATION_OUTPUT_PATH",
            str(tmp_path / "historical_recommendations.csv"),
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RANKING_OUTPUT_PATH",
            str(future_ranking_path),
        ),
        patch.object(
            causal_recommendations,
            "DEFAULT_FUTURE_RECOMMENDATION_OUTPUT_PATH",
            str(future_recommendation_path),
        ),
        patch.object(causal_recommendations, "DEFAULT_METRICS_OUTPUT_PATH", str(metrics_path)),
    ):
        result = run_all_causal_adjusted_recommendations(top_n=2)

    assert result["historical"]["status"] == "skipped"
    assert future_ranking_path.exists()
    assert future_recommendation_path.exists()
    assert metrics_path.exists()


def sample_average_rankings() -> pd.DataFrame:
    hourly = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h"),
            "window": ["test"] * 3,
            "model": ["model_a"] * 3,
            "decision_date": ["2026-01-01"] * 3,
            "actual_price_eur_mwh": [1.0, 1.0, 1.0],
            "predicted_price_eur_mwh": [1.0, 1.0, 1.0],
            "previous_day_price_eur_mwh": [1.0, 1.0, 1.0],
            "actual_carbon_intensity_g_co2e_per_kwh": [1.0, 2.0, 3.0],
            "predicted_carbon_intensity_g_co2e_per_kwh": [1.0, 2.0, 3.0],
            "actual_total_emissions_kg_co2e": [1.0, 2.0, 3.0],
            "predicted_total_emissions_kg_co2e": [1.0, 2.0, 3.0],
        }
    )
    rankings = build_workload_decision_rankings(
        hourly,
        WorkloadConstraints(price_weight=0.0, carbon_weight=1.0),
    )
    rankings["ranking_model_score"] = 1 - rankings["predicted_combined_score"]
    rankings["ranking_model_decision_score"] = rankings["predicted_combined_score"]
    rankings["ranking_model_score_source"] = "test"
    return rankings


def sample_marginal_proxy() -> pd.DataFrame:
    rows = []
    for basis in ("actual", "predicted"):
        for timestamp, intensity in zip(
            pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h"),
            [30.0, 20.0, 10.0],
            strict=True,
        ):
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "methodology": "direct_operational_emissions",
                    "window": "test",
                    "model": "model_a",
                    "basis": basis,
                    "marginal_carbon_intensity_g_co2e_per_kwh": intensity,
                    "marginal_source": "gas",
                    "marginal_proxy_confidence": "high",
                }
            )
    return pd.DataFrame(rows)
