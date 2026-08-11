from __future__ import annotations

from scripts.gated_retrain import evaluate_promotion


def champion_payload(model: str, carbon_mae: float, carbon_regret: float) -> dict[str, object]:
    return {
        "champion_model": model,
        "weights": {
            "carbon_intensity_error": 0.45,
            "carbon_regret": 0.25,
            "top_5_ranking_loss": 0.2,
            "price_direction_error": 0.1,
        },
        "models": [
            {
                "model": model,
                "carbon_intensity_mae_g_co2e_per_kwh": carbon_mae,
                "carbon_regret_g_co2e_per_kwh": carbon_regret,
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
