from __future__ import annotations

import json

from scripts.write_orchestration_report import main


def test_write_orchestration_report_records_retrain_and_state_source(tmp_path) -> None:
    output_path = tmp_path / "orchestration.json"

    result = main(
        [
            "--output-path",
            str(output_path),
            "--event-name",
            "schedule",
            "--run-id",
            "123",
            "--run-attempt",
            "2",
            "--preflight-result",
            "success",
            "--retrain-requested",
            "true",
            "--retrain-result",
            "success",
            "--operational-action",
            "retrain",
            "--operational-reasons",
            "forecast_monitoring_retraining_recommended,required_model_artifacts_missing",
            "--state-source",
            "retrained-operational-state",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["preflight"]["retrain_requested"] is True
    assert report["preflight"]["reasons"] == [
        "forecast_monitoring_retraining_recommended",
        "required_model_artifacts_missing",
    ]
    assert report["publish"]["state_source"] == "retrained-operational-state"
