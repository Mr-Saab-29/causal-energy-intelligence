from __future__ import annotations

import os

from scripts.gated_retrain import (
    evaluate_operational_evidence,
    evaluate_promotion,
    missing_quantile_artifacts,
    prune_snapshots,
)


def champion_payload(model: str, carbon_mae: float, carbon_regret: float) -> dict[str, object]:
    return {
        "champion_model": model,
        "weights": {
            "recommendation_regret": 0.35,
            "carbon_intensity_error": 0.1,
            "carbon_regret": 0.25,
            "top_5_ranking_loss": 0.2,
            "price_direction_error": 0.1,
        },
        "models": [
            {
                "model": model,
                "carbon_intensity_mae_g_co2e_per_kwh": carbon_mae,
                "carbon_regret_g_co2e_per_kwh": carbon_regret,
                "mean_top_1_combined_regret": carbon_regret,
                "top_5_ranking_loss": 0.2,
                "price_direction_error": 0.1,
            }
        ],
    }


def test_evaluate_promotion_accepts_better_candidate() -> None:
    incumbent = champion_payload("hist_gradient_boosting", carbon_mae=1.0, carbon_regret=0.5)
    candidate = champion_payload("lightgbm", carbon_mae=0.8, carbon_regret=0.4)

    decision = evaluate_promotion(incumbent, candidate, min_improvement=0.0)

    assert decision["promoted"] is True
    assert decision["promotion_score_vs_incumbent"] < 1


def test_evaluate_promotion_rejects_worse_candidate() -> None:
    incumbent = champion_payload("hist_gradient_boosting", carbon_mae=1.0, carbon_regret=0.5)
    candidate = champion_payload("lightgbm", carbon_mae=1.2, carbon_regret=0.7)

    decision = evaluate_promotion(incumbent, candidate, min_improvement=0.0)

    assert decision["promoted"] is False
    assert decision["promotion_score_vs_incumbent"] > 1


def test_evaluate_promotion_rejects_guarded_metric_regression() -> None:
    incumbent = champion_payload("hist_gradient_boosting", carbon_mae=10.0, carbon_regret=1.0)
    candidate = champion_payload("lightgbm", carbon_mae=1.0, carbon_regret=1.2)

    decision = evaluate_promotion(incumbent, candidate, min_improvement=0.0)

    assert decision["promoted"] is False
    assert "carbon_regret_g_co2e_per_kwh" in decision["guarded_metric_degradations"]


def test_evaluate_promotion_ignores_operational_metrics_with_small_sample() -> None:
    incumbent = champion_payload("hist_gradient_boosting", carbon_mae=1.0, carbon_regret=0.5)
    candidate = champion_payload("lightgbm", carbon_mae=0.8, carbon_regret=0.4)
    operational = {
        "available": True,
        "decision_groups": 2,
        "rows": 10,
        "top_5_hit_rate": 0.0,
        "mean_carbon_regret_g_co2e_per_kwh": 99.0,
    }

    decision = evaluate_promotion(
        incumbent,
        candidate,
        min_improvement=0.0,
        operational_outcome=operational,
        min_operational_decision_groups=7,
    )

    assert decision["promoted"] is True
    assert decision["operational_evidence"]["status"] == "insufficient_history"
    assert decision["operational_evidence"]["blocks_promotion"] is False


def test_evaluate_promotion_blocks_on_poor_operational_metrics_with_enough_history() -> None:
    incumbent = champion_payload("hist_gradient_boosting", carbon_mae=1.0, carbon_regret=0.5)
    candidate = champion_payload("lightgbm", carbon_mae=0.8, carbon_regret=0.4)
    operational = {
        "available": True,
        "decision_groups": 8,
        "rows": 40,
        "top_5_hit_rate": 0.4,
        "mean_carbon_regret_g_co2e_per_kwh": 3.0,
    }

    decision = evaluate_promotion(
        incumbent,
        candidate,
        min_improvement=0.0,
        operational_outcome=operational,
        min_operational_decision_groups=7,
    )

    assert decision["promoted"] is False
    assert decision["operational_evidence"]["status"] == "guarded_fail"
    assert decision["operational_evidence"]["blocks_promotion"] is True
    assert "top_5_hit_rate_below_floor" in decision["operational_evidence"]["failures"]


def test_evaluate_operational_evidence_reports_unavailable_metrics() -> None:
    evidence = evaluate_operational_evidence({}, min_decision_groups=7)

    assert evidence["status"] == "unavailable"
    assert evidence["blocks_promotion"] is False


def test_prune_snapshots_keeps_newest_directories(tmp_path) -> None:
    old_snapshot = tmp_path / "old"
    middle_snapshot = tmp_path / "middle"
    new_snapshot = tmp_path / "new"
    for offset, snapshot in enumerate([old_snapshot, middle_snapshot, new_snapshot]):
        snapshot.mkdir()
        timestamp = 1_700_000_000 + offset
        os.utime(snapshot, (timestamp, timestamp))

    prune_snapshots(tmp_path, keep=2)

    assert not old_snapshot.exists()
    assert middle_snapshot.exists()
    assert new_snapshot.exists()


def test_missing_quantile_artifacts_requires_every_operational_target(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ridge_price_quantile.joblib").touch()

    missing = missing_quantile_artifacts(tmp_path)

    assert "price" not in missing
    assert "consumption" in missing
    assert "production" in missing
