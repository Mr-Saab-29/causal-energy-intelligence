"""Versioned causal DAGs for grid effects and product effectiveness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class CausalNode:
    """A named causal variable with its analytical role and measurement timing."""

    name: str
    role: str
    timing: str


@dataclass(frozen=True)
class CausalDag:
    """Serializable directed acyclic graph specification."""

    name: str
    version: str
    treatment: str
    outcome: str
    nodes: tuple[CausalNode, ...]
    edges: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }


PRE_DECISION_VARIABLES = (
    "calendar",
    "weather_forecast",
    "baseline_load_forecast",
    "renewable_generation_forecast",
    "known_generation_outages",
    "fuel_price",
    "carbon_price",
    "interconnector_capacity",
    "initial_storage_state",
)

GRID_RESPONSE_VARIABLES = (
    "generator_dispatch",
    "cross_border_flows",
    "storage_dispatch",
    "renewable_curtailment",
    "balancing_activation",
)


def get_grid_effect_dag() -> CausalDag:
    """Return the DAG for the total grid-emissions effect of workload timing."""
    nodes = tuple(
        CausalNode(name, "pre_treatment_covariate", "pre_treatment")
        for name in PRE_DECISION_VARIABLES
    ) + (
        CausalNode("baseline_grid_state", "derived_confounder", "pre_treatment"),
        CausalNode("workload_timing", "treatment", "treatment"),
        CausalNode("net_load", "mediator", "post_treatment"),
        *(CausalNode(name, "mediator", "post_treatment") for name in GRID_RESPONSE_VARIABLES),
        CausalNode("total_operational_emissions", "outcome", "post_treatment"),
    )
    edges = tuple((name, "baseline_grid_state") for name in PRE_DECISION_VARIABLES)
    edges += (
        ("baseline_grid_state", "workload_timing"),
        ("workload_timing", "net_load"),
    )
    edges += tuple(("baseline_grid_state", name) for name in GRID_RESPONSE_VARIABLES)
    edges += tuple(("net_load", name) for name in GRID_RESPONSE_VARIABLES)
    edges += tuple((name, "total_operational_emissions") for name in GRID_RESPONSE_VARIABLES)
    return CausalDag(
        name="short_run_workload_shift_grid_effect",
        version="grid_effect_v1",
        treatment="workload_timing",
        outcome="total_operational_emissions",
        nodes=nodes,
        edges=edges,
    )


def get_product_effect_dag() -> CausalDag:
    """Return the separate DAG for the effect of showing a recommendation."""
    return CausalDag(
        name="recommendation_product_effect",
        version="product_effect_v1",
        treatment="recommendation_shown",
        outcome="total_operational_emissions",
        nodes=(
            CausalNode("user_constraints", "pre_treatment_covariate", "pre_treatment"),
            CausalNode("recommendation_shown", "treatment", "treatment"),
            CausalNode("recommendation_accepted", "mediator", "post_treatment"),
            CausalNode("realized_workload_shift", "mediator", "post_treatment"),
            CausalNode("grid_response", "mediator", "post_treatment"),
            CausalNode("total_operational_emissions", "outcome", "post_treatment"),
        ),
        edges=(
            ("user_constraints", "recommendation_accepted"),
            ("user_constraints", "realized_workload_shift"),
            ("recommendation_shown", "recommendation_accepted"),
            ("recommendation_accepted", "realized_workload_shift"),
            ("realized_workload_shift", "grid_response"),
            ("grid_response", "total_operational_emissions"),
        ),
    )


def get_causal_dag_edges() -> list[tuple[str, str]]:
    """Return grid-effect edges for callers using the original helper."""
    return list(get_grid_effect_dag().edges)


def is_acyclic(dag: CausalDag) -> bool:
    """Return whether the graph contains no directed cycle."""
    nodes = {node.name for node in dag.nodes}
    indegree = {node: 0 for node in nodes}
    children = adjacency(dag.edges)
    for source, target in dag.edges:
        if source not in nodes or target not in nodes:
            return False
        indegree[target] += 1
    pending = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        for child in children.get(node, set()):
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    return visited == len(nodes)


def has_directed_path(dag: CausalDag, source: str, target: str) -> bool:
    """Return whether target is reachable from source."""
    return target in descendants(dag, source)


def descendants(dag: CausalDag, source: str) -> set[str]:
    """Return all directed descendants of source."""
    children = adjacency(dag.edges)
    found: set[str] = set()
    pending = list(children.get(source, set()))
    while pending:
        node = pending.pop()
        if node in found:
            continue
        found.add(node)
        pending.extend(children.get(node, set()))
    return found


def adjacency(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Build a child lookup from directed edges."""
    children: dict[str, set[str]] = {}
    for source, target in edges:
        children.setdefault(source, set()).add(target)
    return children
