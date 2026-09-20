from __future__ import annotations

import copy
import json

import pytest

from src.causal.dag import (
    get_grid_effect_dag,
    get_product_effect_dag,
    has_directed_path,
    is_acyclic,
)
from src.causal.estimand import (
    build_causal_contract,
    load_estimand_spec,
    resolve_baseline_timestamp,
    validate_estimand_spec,
)
from src.causal.estimate_effect import estimate_treatment_effect


def test_default_estimand_and_dags_are_valid() -> None:
    spec = load_estimand_spec()
    grid_dag = get_grid_effect_dag()
    product_dag = get_product_effect_dag()

    assert validate_estimand_spec(spec, grid_dag) == []
    assert is_acyclic(grid_dag)
    assert is_acyclic(product_dag)
    assert has_directed_path(grid_dag, "workload_timing", "total_operational_emissions")
    assert has_directed_path(
        product_dag,
        "recommendation_shown",
        "total_operational_emissions",
    )


def test_baseline_uses_planned_start_override_when_provided() -> None:
    baseline = resolve_baseline_timestamp(
        dashboard_access_time="2026-09-18T08:00:00+02:00",
        user_planned_start_time="2026-09-18T12:00:00+02:00",
    )

    assert baseline == {
        "source": "user_planned_start_time",
        "timestamp_utc": "2026-09-18T10:00:00+00:00",
    }


def test_baseline_falls_back_to_dashboard_access_time() -> None:
    baseline = resolve_baseline_timestamp("2026-09-18T08:00:00+02:00")

    assert baseline == {
        "source": "dashboard_access_time",
        "timestamp_utc": "2026-09-18T06:00:00+00:00",
    }


def test_baseline_rejects_timezone_naive_input() -> None:
    with pytest.raises(ValueError, match="timezone offset"):
        resolve_baseline_timestamp("2026-09-18T08:00:00")


def test_validation_rejects_post_treatment_adjustment() -> None:
    spec = copy.deepcopy(load_estimand_spec())
    spec["adjustment_set"].append("generator_dispatch")

    errors = validate_estimand_spec(spec)

    assert "adjustment_variable_not_pre_treatment:generator_dispatch" in errors
    assert "post_treatment_adjustment_forbidden:generator_dispatch" in errors


def test_contract_writer_emits_strict_versioned_json(tmp_path) -> None:
    output_path = tmp_path / "estimand_spec.json"

    payload = build_causal_contract(output_path=output_path)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["validation"]["status"] == "ok"
    assert persisted["estimand"]["estimand_version"] == "short_run_workload_shift_v1"
    assert persisted["estimand"]["target_population"]["primary_workload_profiles"] == [
        "data_center_batch",
        "ev_battery_charging",
    ]
    assert "NaN" not in output_path.read_text(encoding="utf-8")


def test_effect_estimation_refuses_a_different_causal_question() -> None:
    with pytest.raises(ValueError, match="does not match"):
        estimate_treatment_effect("recommendation_shown", "total_operational_emissions", [])


def test_effect_estimation_reports_that_identified_estimator_is_pending() -> None:
    result = estimate_treatment_effect("workload_timing", "total_operational_emissions", [])

    assert result["status"] == "not_estimated"
    assert result["effect"] is None
    assert result["estimand_version"] == "short_run_workload_shift_v1"
