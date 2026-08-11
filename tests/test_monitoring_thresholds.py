from __future__ import annotations

from src.monitoring.forecast_monitor import load_monitoring_thresholds


def test_load_monitoring_thresholds_merges_defaults(tmp_path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text("degradation_ratio: 1.5\n", encoding="utf-8")

    thresholds = load_monitoring_thresholds(path)

    assert thresholds["degradation_ratio"] == 1.5
    assert thresholds["recent_window_days"] == 14
