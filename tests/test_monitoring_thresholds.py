from __future__ import annotations

from src.monitoring.forecast_monitor import evaluate_retraining_trigger, load_monitoring_thresholds


def test_load_monitoring_thresholds_merges_defaults(tmp_path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text("degradation_ratio: 1.5\n", encoding="utf-8")

    thresholds = load_monitoring_thresholds(path)

    assert thresholds["degradation_ratio"] == 1.5
    assert thresholds["recent_window_days"] == 14
    assert thresholds["max_high_uncertainty_share"] == 0.30


def test_evaluate_retraining_trigger_flags_recommendation_drift() -> None:
    trigger = evaluate_retraining_trigger(
        pipeline_health={"status": "pass"},
        historical={"available": False, "reason": "historical unavailable"},
        operational={"available": False, "reason": "operational unavailable"},
        source_drift={"available": False, "reason": "source unavailable"},
        recommendation_drift={
            "available": True,
            "high_uncertainty_share": 0.6,
            "average_confidence_score": 0.3,
            "recommendation_status_counts": {
                "no_low_risk_recommendation_available": 3,
            },
            "rows": 10,
            "rank_overlap_with_previous": 0.2,
        },
        references={},
        thresholds={
            "min_operational_rows": 12,
            "degradation_ratio": 1.25,
            "top5_drop_threshold": 0.15,
            "min_price_direction_accuracy": 0.50,
            "source_smape_degradation_ratio": 1.25,
            "max_high_uncertainty_share": 0.30,
            "max_no_low_risk_recommendation_share": 0.10,
            "min_average_confidence_score": 0.50,
            "min_rank_overlap_with_previous": 0.50,
        },
    )

    assert trigger["retraining_recommended"] is True
    assert "recommendation_high_uncertainty_share_high" in trigger["reasons"]
    assert "recommendation_no_low_risk_share_high" in trigger["reasons"]
    assert "recommendation_average_confidence_low" in trigger["reasons"]
    assert "recommendation_rank_overlap_low" in trigger["reasons"]
