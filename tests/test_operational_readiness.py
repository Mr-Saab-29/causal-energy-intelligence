from __future__ import annotations

import pandas as pd

from src.monitoring.operational_readiness import build_operational_audit_readiness


def test_operational_readiness_passes_with_current_month_actuals_and_history(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-09-07T22:00:00+00:00",
                "2026-09-08T00:00:00+00:00",
            ],
            "price_eur_mwh": [50.0, 45.0],
        }
    )
    recommendations = pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-09-07T22:00:00+00:00",
                "2026-09-09T04:00:00+00:00",
            ],
            "decision_group": ["2026-09-07", "2026-09-09"],
            "model": ["model_a", "model_a"],
        }
    )
    features_path = tmp_path / "features.csv"
    recommendations_path = tmp_path / "recommendations.csv"
    features.to_csv(features_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    report = build_operational_audit_readiness(
        features_path=features_path,
        recommendation_history_path=recommendations_path,
        output_path=tmp_path / "readiness.json",
        as_of_utc="2026-09-09T06:00:00Z",
    )

    assert report["status"] == "pass"
    assert report["checks"]["current_month_actuals_available"] is True
    assert report["checks"]["current_month_recommendations_available"] is True
    assert report["checks"]["settled_current_month_recommendations_available"] is True


def test_operational_readiness_fails_without_current_month_actuals(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp_utc": ["2026-08-24T12:00:00+00:00"],
            "price_eur_mwh": [50.0],
        }
    )
    recommendations = pd.DataFrame(
        {
            "timestamp_utc": ["2026-09-08T12:00:00+00:00"],
            "decision_group": ["2026-09-08"],
            "model": ["model_a"],
        }
    )
    features_path = tmp_path / "features.csv"
    recommendations_path = tmp_path / "recommendations.csv"
    features.to_csv(features_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    report = build_operational_audit_readiness(
        features_path=features_path,
        recommendation_history_path=recommendations_path,
        output_path=tmp_path / "readiness.json",
        as_of_utc="2026-09-09T06:00:00Z",
    )

    assert report["status"] == "fail"
    assert "current_month_actuals_missing" in report["reasons"]


def test_operational_readiness_fails_without_current_month_recommendations(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "timestamp_utc": ["2026-09-08T00:00:00+00:00"],
            "price_eur_mwh": [45.0],
        }
    )
    recommendations = pd.DataFrame(
        {
            "timestamp_utc": ["2026-08-12T09:00:00+00:00"],
            "decision_group": ["2026-08-12"],
            "model": ["model_a"],
        }
    )
    features_path = tmp_path / "features.csv"
    recommendations_path = tmp_path / "recommendations.csv"
    features.to_csv(features_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    report = build_operational_audit_readiness(
        features_path=features_path,
        recommendation_history_path=recommendations_path,
        output_path=tmp_path / "readiness.json",
        as_of_utc="2026-09-09T06:00:00Z",
    )

    assert report["status"] == "fail"
    assert "current_month_operational_recommendations_missing" in report["reasons"]


def test_operational_readiness_fails_when_current_month_recommendations_are_unsettled(
    tmp_path,
) -> None:
    features = pd.DataFrame(
        {
            "timestamp_utc": ["2026-09-08T00:00:00+00:00"],
            "price_eur_mwh": [45.0],
        }
    )
    recommendations = pd.DataFrame(
        {
            "timestamp_utc": ["2026-09-09T04:00:00+00:00"],
            "decision_group": ["2026-09-09"],
            "model": ["model_a"],
        }
    )
    features_path = tmp_path / "features.csv"
    recommendations_path = tmp_path / "recommendations.csv"
    features.to_csv(features_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    report = build_operational_audit_readiness(
        features_path=features_path,
        recommendation_history_path=recommendations_path,
        output_path=tmp_path / "readiness.json",
        as_of_utc="2026-09-09T06:00:00Z",
    )

    assert report["status"] == "fail"
    assert "current_month_recommendations_not_settled_yet" in report["reasons"]
