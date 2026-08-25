from __future__ import annotations

import os

from scripts.gated_retrain import evaluate_promotion, prune_snapshots


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
