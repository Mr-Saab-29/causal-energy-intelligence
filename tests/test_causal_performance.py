import pandas as pd

from src.monitoring.causal_performance import build_causal_performance_report


def estimator_report(value: float) -> dict:
    return {
        "status": "ok",
        "horizons": [
            {
                "horizon_hours": 6,
                "point_estimates": {
                    "mean_marginal_response_kgco2e_per_mwh": value,
                },
            }
        ],
    }


def test_causal_monitor_passes_with_settled_regret_and_stable_estimate() -> None:
    audit = pd.DataFrame(
        {
            "recommendation_basis": ["causal_adjusted_mvp"] * 14,
            "recommendation_rank": [1] * 14,
            "decision_group": [f"2026-09-{day:02d}" for day in range(1, 15)],
            "carbon_regret_g_co2e_per_kwh_observed": [2.0] * 14,
        }
    )
    history = pd.DataFrame(
        {
            "generated_at_utc": ["2026-09-01T00:00:00+00:00"],
            "estimator": ["constrained_regression"],
            "horizon_hours": [6],
            "estimate_kgco2e_per_mwh": [100.0],
        }
    )

    report, _ = build_causal_performance_report(
        audit,
        [("constrained_regression", estimator_report(110.0))],
        history,
        generated_at_utc="2026-10-01T00:00:00+00:00",
    )

    assert report["status"] == "pass"
    assert report["realized_causal_regret"]["status"] == "pass"
    assert report["estimator_drift"]["comparisons"][0]["relative_drift"] == 0.1


def test_causal_monitor_does_not_use_average_basis_as_causal_evidence() -> None:
    audit = pd.DataFrame(
        {
            "recommendation_basis": ["average_carbon"],
            "recommendation_rank": [1],
            "decision_group": ["2026-09-01"],
            "carbon_regret_g_co2e_per_kwh_observed": [0.0],
        }
    )

    report, history = build_causal_performance_report(audit, [], pd.DataFrame())

    assert report["realized_causal_regret"]["settled_decision_groups"] == 0
    assert report["status"] == "insufficient_history"
    assert list(history.columns) == [
        "generated_at_utc",
        "estimator",
        "horizon_hours",
        "estimate_kgco2e_per_mwh",
    ]
