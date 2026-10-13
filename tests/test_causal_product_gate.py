from src.causal.product_gate import evaluate_product_gate


def diagnostics(origin: str = "observed", status: str = "pass") -> dict:
    return {
        "data_origin": origin,
        "diagnostics_ready": True,
        "causal_claim_eligible": origin == "observed" and status == "pass",
        "horizons": [
            {
                "treatment_overlap_and_positivity": {"status": status},
                "time_series_uncertainty": {"status": status},
            }
        ],
    }


def test_product_gate_does_not_promote_synthetic_evidence() -> None:
    report = evaluate_product_gate(
        diagnostics("synthetic_product_bootstrap"),
        {
            "structural_validation_ready": True,
            "data_origin": "synthetic_dispatch",
            "causal_claim_eligible": False,
            "model_risk": "low",
        },
        {},
    )

    assert report["promoted"] is False
    assert report["basis_label"] == "Causal-adjusted MVP"
    assert report["evidence_tier"] == "marginal_proxy_mvp"
    assert report["recommendations_allowed"] is True


def test_product_gate_promotes_only_complete_operational_evidence() -> None:
    report = evaluate_product_gate(
        diagnostics(),
        {
            "structural_validation_ready": True,
            "data_origin": "pypsa_eur",
            "causal_claim_eligible": True,
            "model_risk": "low",
        },
        {
            "status": "pass",
            "realized_causal_regret": {"status": "pass"},
            "estimator_drift": {"status": "pass"},
        },
        {"future": {"method": "identified_causal_effect"}},
    )

    assert report["promoted"] is True
    assert report["basis_label"] == "Causal-adjusted"
    assert report["evidence_tier"] == "operationally_validated"


def test_product_gate_suppresses_identified_model_with_poor_overlap() -> None:
    report = evaluate_product_gate(diagnostics(status="fail"), {}, {})

    assert report["recommendations_allowed"] is False
    assert "poor_treatment_overlap" in report["suppression_reasons"]
