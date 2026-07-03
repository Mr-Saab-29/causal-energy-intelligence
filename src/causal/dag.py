"""Causal DAG definition for energy workload shifting decisions."""

from __future__ import annotations


def get_causal_dag_edges() -> list[tuple[str, str]]:
    """Return the baseline causal graph as directed edges."""
    return [
        ("weather", "electricity_demand"),
        ("electricity_demand", "electricity_price"),
        ("renewable_generation", "carbon_intensity"),
        ("carbon_intensity", "workload_shift_decision"),
        ("electricity_price", "workload_shift_decision"),
    ]

