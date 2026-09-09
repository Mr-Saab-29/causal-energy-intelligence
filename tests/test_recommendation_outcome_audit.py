from __future__ import annotations

import pandas as pd

from src.monitoring import recommendation_outcome_audit as audit


def test_recommendation_outcome_audit_compares_settled_predictions_to_actuals(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "load_emission_factor_config",
        lambda _: {
            "direct_operational_emissions": {
                "nuclear": 1.0,
                "gas": 10.0,
                "coal": 20.0,
                "oil": 15.0,
                "wind": 0.0,
                "solar": 0.0,
                "hydro": 0.0,
                "bioenergy": 5.0,
            }
        },
    )
    timestamps = pd.date_range("2026-09-01T00:00:00Z", periods=3, freq="h")
    features = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_mwh": [30.0, 20.0, 40.0],
            "consumption_mwh": [100.0, 110.0, 120.0],
            "total_production_mwh": [100.0, 100.0, 100.0],
            "nuclear_mwh": [70.0, 80.0, 60.0],
            "gas_mwh": [20.0, 10.0, 30.0],
            "coal_mwh": [0.0, 0.0, 0.0],
            "oil_mwh": [0.0, 0.0, 0.0],
            "wind_mwh": [10.0, 10.0, 10.0],
            "solar_mwh": [0.0, 0.0, 0.0],
            "hydro_mwh": [0.0, 0.0, 0.0],
            "bioenergy_mwh": [0.0, 0.0, 0.0],
        }
    )
    rankings = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "forecast_generated_at_utc": ["2026-08-31T12:00:00+00:00"] * 3,
            "window": ["future_24h"] * 3,
            "model": ["model_a"] * 3,
            "decision_group": ["2026-09-01"] * 3,
            "predicted_decision_rank": [1, 2, 3],
            "predicted_avg_price_eur_mwh": [22.0, 25.0, 45.0],
            "predicted_avg_carbon_intensity_g_co2e_per_kwh": [3.0, 2.0, 4.0],
            "predicted_total_emissions_kg_co2e": [300.0, 200.0, 400.0],
            "predicted_consumption_mwh": [90.0, 115.0, 125.0],
            "predicted_total_production_mwh": [95.0, 105.0, 125.0],
            "previous_day_price_eur_mwh_observed": [None, None, None],
        }
    )
    recommendations = rankings.iloc[:2].copy()
    recommendations["recommendation_rank"] = [1, 2]
    recommendations["recommendation_status"] = "recommended"
    recommendations["confidence_score"] = [0.8, 0.7]
    features_path = tmp_path / "features.csv"
    rankings_path = tmp_path / "rankings.csv"
    recommendations_path = tmp_path / "recommendations.csv"
    output_path = tmp_path / "audit.csv"
    metrics_path = tmp_path / "metrics.json"
    features.to_csv(features_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    summary = audit.build_recommendation_outcome_audit(
        features_path=features_path,
        recommendation_history_path=recommendations_path,
        ranking_history_path=rankings_path,
        output_path=output_path,
        metrics_output_path=metrics_path,
    )

    output = pd.read_csv(output_path)
    assert summary["available"] is True
    assert summary["rows"] == 2
    assert output["recommendation_rank"].tolist() == [1, 2]
    assert output.loc[0, "price_error"] == -8.0
    assert output.loc[0, "consumption_error"] == -10.0
    assert output.loc[0, "actual_decision_rank_observed"] == 2
    assert summary["top_1_hit_rate"] == 0.0


def test_recommendation_outcome_audit_reports_missing_inputs(tmp_path) -> None:
    summary = audit.build_recommendation_outcome_audit(
        features_path=tmp_path / "missing_features.csv",
        recommendation_history_path=tmp_path / "missing_recommendations.csv",
        ranking_history_path=tmp_path / "missing_rankings.csv",
        output_path=tmp_path / "audit.csv",
        metrics_output_path=tmp_path / "metrics.json",
    )

    assert summary["available"] is False
    assert "actual feature rows" in summary["reason"]
