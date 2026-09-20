"""Causal estimand contract, validation, and artifact generation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.causal.dag import (
    CausalDag,
    descendants,
    get_grid_effect_dag,
    get_product_effect_dag,
    has_directed_path,
    is_acyclic,
)

DEFAULT_ESTIMAND_CONFIG_PATH = "config/causal_estimand.json"
DEFAULT_ESTIMAND_OUTPUT_PATH = "reports/causal/estimand_spec.json"


def load_estimand_spec(path: str | Path = DEFAULT_ESTIMAND_CONFIG_PATH) -> dict[str, Any]:
    """Load the versioned estimand definition."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_baseline_timestamp(
    dashboard_access_time: str | datetime,
    user_planned_start_time: str | datetime | None = None,
) -> dict[str, str]:
    """Resolve the user override or dashboard-access fallback as an exact UTC baseline."""
    source = (
        "user_planned_start_time"
        if user_planned_start_time is not None
        else "dashboard_access_time"
    )
    value = (
        user_planned_start_time
        if user_planned_start_time is not None
        else dashboard_access_time
    )
    timestamp = parse_aware_datetime(value)
    return {
        "source": source,
        "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat(),
    }


def validate_estimand_spec(spec: dict[str, Any], dag: CausalDag | None = None) -> list[str]:
    """Return contract errors, including unsafe post-treatment adjustment."""
    dag = dag or get_grid_effect_dag()
    errors: list[str] = []
    treatment = spec.get("treatment", {}).get("graph_node")
    outcome = spec.get("outcome", {}).get("graph_node")
    node_by_name = {node.name: node for node in dag.nodes}

    if not spec.get("estimand_version"):
        errors.append("estimand_version_missing")
    if treatment != dag.treatment:
        errors.append("treatment_does_not_match_dag")
    if outcome != dag.outcome:
        errors.append("outcome_does_not_match_dag")
    if not is_acyclic(dag):
        errors.append("dag_is_not_acyclic")
    if treatment and outcome and not has_directed_path(dag, treatment, outcome):
        errors.append("treatment_has_no_directed_path_to_outcome")

    treatment_descendants = descendants(dag, treatment) if treatment in node_by_name else set()
    mediators = set(spec.get("mediators", []))
    for variable in spec.get("adjustment_set", []):
        node = node_by_name.get(variable)
        if node is None:
            errors.append(f"adjustment_variable_missing_from_dag:{variable}")
            continue
        if node.timing != "pre_treatment":
            errors.append(f"adjustment_variable_not_pre_treatment:{variable}")
        if variable in treatment_descendants or variable in mediators:
            errors.append(f"post_treatment_adjustment_forbidden:{variable}")

    treatment_spec = spec.get("treatment", {})
    if float(treatment_spec.get("energy_mwh", 0)) <= 0:
        errors.append("treatment_energy_must_be_positive")
    if int(treatment_spec.get("duration_hours", 0)) <= 0:
        errors.append("treatment_duration_must_be_positive")
    if not treatment_spec.get("energy_conservation_required"):
        errors.append("paired_shift_must_conserve_energy")

    baseline = spec.get("baseline", {})
    if baseline.get("default_source") != "dashboard_access_time":
        errors.append("baseline_default_must_be_dashboard_access_time")
    if baseline.get("optional_override") != "user_planned_start_time":
        errors.append("baseline_override_must_be_user_planned_start_time")

    horizons = spec.get("time_horizon", {}).get("sensitivity_response_horizon_hours", [])
    if not horizons or any(int(value) <= 0 for value in horizons):
        errors.append("positive_response_horizons_required")
    return sorted(set(errors))


def build_causal_contract(
    config_path: str | Path = DEFAULT_ESTIMAND_CONFIG_PATH,
    output_path: str | Path = DEFAULT_ESTIMAND_OUTPUT_PATH,
) -> dict[str, Any]:
    """Validate and write the estimand plus grid and product DAGs as strict JSON."""
    spec = load_estimand_spec(config_path)
    grid_dag = get_grid_effect_dag()
    product_dag = get_product_effect_dag()
    errors = validate_estimand_spec(spec, grid_dag)
    if not is_acyclic(product_dag):
        errors.append("product_dag_is_not_acyclic")
    if not has_directed_path(product_dag, product_dag.treatment, product_dag.outcome):
        errors.append("product_dag_treatment_has_no_directed_path_to_outcome")
    if errors:
        raise ValueError(f"Invalid causal estimand contract: {', '.join(errors)}")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from": str(config_path),
        "estimand": spec,
        "dags": {
            "grid_effect": grid_dag.to_dict(),
            "product_effect": product_dag.to_dict(),
        },
        "validation": {
            "status": "ok",
            "errors": [],
            "checks": [
                "dag_acyclic",
                "treatment_to_outcome_path",
                "adjustment_variables_pre_treatment",
                "no_treatment_descendants_in_adjustment_set",
                "paired_shift_energy_conservation",
                "baseline_precedence",
            ],
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def parse_aware_datetime(value: str | datetime) -> datetime:
    """Parse a timestamp and reject ambiguous timezone-naive values."""
    timestamp = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
    )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Baseline timestamps must include a timezone offset.")
    return timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and write the causal estimand contract.")
    parser.add_argument("--config-path", default=DEFAULT_ESTIMAND_CONFIG_PATH)
    parser.add_argument("--output-path", default=DEFAULT_ESTIMAND_OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_causal_contract(args.config_path, args.output_path)
    print(
        json.dumps(
            {
                "status": payload["validation"]["status"],
                "estimand_version": payload["estimand"]["estimand_version"],
                "output": args.output_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
