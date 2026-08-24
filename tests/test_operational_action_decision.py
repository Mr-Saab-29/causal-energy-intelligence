from __future__ import annotations

import json

import scripts.decide_operational_action as decision
from src.models.baseline_price import PRODUCTION_SIGNAL_TARGETS


def test_decision_recommends_when_monitor_clean_and_models_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(decision, "ROOT", tmp_path)
    monitor_path = tmp_path / "reports/metrics/forecast_monitoring.json"
    monkeypatch.setattr(decision, "FORECAST_MONITORING_PATH", monitor_path)
    monitor_path.parent.mkdir(parents=True)
    monitor_path.write_text(json.dumps({"retraining_recommended": False}), encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for target in ["consumption", *PRODUCTION_SIGNAL_TARGETS, "price"]:
        (model_dir / f"model_{target}_baseline.joblib").touch()

    result = decision.decide_operational_action()

    assert result["action"] == "recommend"
    assert result["retrain"] is False
    assert result["reasons"] == []


def test_decision_retrains_when_monitor_recommends_retraining(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(decision, "ROOT", tmp_path)
    monitor_path = tmp_path / "reports/metrics/forecast_monitoring.json"
    monkeypatch.setattr(decision, "FORECAST_MONITORING_PATH", monitor_path)
    monitor_path.parent.mkdir(parents=True)
    monitor_path.write_text(json.dumps({"retraining_recommended": True}), encoding="utf-8")

    result = decision.decide_operational_action()

    assert result["action"] == "retrain"
    assert "forecast_monitoring_retraining_recommended" in result["reasons"]
