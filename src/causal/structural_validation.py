"""Structural validation against paired European dispatch counterfactuals."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.carbon.intensity import load_emission_factor_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/causal_structural_validation.json"
DEFAULT_FACTORS_PATH = ROOT / "config/emission_factors.yaml"
DEFAULT_PYPSA_MANIFEST_PATH = (
    ROOT / "data/external/pypsa_eur_counterfactual_manifest.json"
)
DEFAULT_EXTERNAL_ESTIMATES_PATH = (
    ROOT / "data/external/structural_dispatch_counterfactuals.csv"
)
DEFAULT_SYNTHETIC_DATASET_PATH = (
    ROOT / "reports/demo/causal/observed_treatment_estimator_dataset.parquet"
)
DEFAULT_OBSERVED_CONSTRAINED_PATH = (
    ROOT / "reports/causal/marginal_response_estimates.csv"
)
DEFAULT_OBSERVED_DML_PATH = ROOT / "reports/causal/dml_marginal_response_estimates.csv"
DEFAULT_SYNTHETIC_CONSTRAINED_PATH = (
    ROOT / "reports/demo/causal/marginal_response_estimates.csv"
)
DEFAULT_SYNTHETIC_DML_PATH = (
    ROOT / "reports/demo/causal/dml_marginal_response_estimates.csv"
)
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_structural_validation.json"
DEFAULT_COMPARISON_PATH = (
    ROOT / "reports/causal/structural_validation_comparison.csv"
)
DEFAULT_STRUCTURAL_ESTIMATES_PATH = (
    ROOT / "reports/causal/structural_dispatch_estimates.csv"
)


def load_validation_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "validation_version",
        "maximum_relative_difference",
        "minimum_dispatch_coverage",
        "minimum_comparison_floor_kgco2e_per_mwh",
        "allow_synthetic_dispatch_fallback",
        "synthetic_storage_response_multiplier",
        "pypsa_carrier_to_emissions_source",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("structural validation config missing: " + ", ".join(missing))
    return config


def dispatch_emissions_delta(
    baseline_dispatch: pd.DataFrame,
    counterfactual_dispatch: pd.DataFrame,
    generator_metadata: pd.DataFrame,
    snapshot_hours: pd.Series,
    emission_factors: dict[str, float],
    carrier_mapping: dict[str, str],
) -> dict[str, float]:
    generators = baseline_dispatch.columns.intersection(counterfactual_dispatch.columns)
    snapshots = baseline_dispatch.index.intersection(counterfactual_dispatch.index)
    baseline = baseline_dispatch.loc[snapshots, generators].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    counterfactual = counterfactual_dispatch.loc[snapshots, generators].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    delta = counterfactual - baseline
    hours = pd.to_numeric(
        snapshot_hours.reindex(snapshots), errors="coerce"
    ).fillna(1.0)
    metadata = generator_metadata.reindex(generators)
    carrier = metadata["carrier"].astype(str)
    source = carrier.map(carrier_mapping)
    factor = source.map(emission_factors)
    mapped = factor.notna()
    delta_energy = delta.mul(hours, axis=0)
    mapped_delta = delta_energy.loc[:, mapped]
    emissions_delta = float(
        mapped_delta.mul(factor[mapped], axis=1).sum().sum()
    )
    absolute_delta = float(delta_energy.abs().sum().sum())
    mapped_absolute_delta = float(mapped_delta.abs().sum().sum())
    coverage = mapped_absolute_delta / absolute_delta if absolute_delta > 0 else 1.0
    return {
        "emissions_delta_kgco2e": emissions_delta,
        "dispatch_mapping_coverage": coverage,
        "absolute_dispatch_delta_mwh": absolute_delta,
        "unmapped_generator_count": int((~mapped).sum()),
    }


def load_pypsa_eur_counterfactuals(
    manifest_path: str | Path,
    config: dict[str, Any],
    emission_factors: dict[str, float],
) -> pd.DataFrame:
    try:
        import pypsa
    except ImportError as exc:
        raise RuntimeError(
            "PyPSA is required to read paired PyPSA-Eur NetCDF networks"
        ) from exc

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for run in manifest.get("runs", []):
        baseline_path = resolve_manifest_path(manifest_path, run["baseline_network"])
        counterfactual_path = resolve_manifest_path(
            manifest_path, run["counterfactual_network"]
        )
        baseline = pypsa.Network(baseline_path)
        counterfactual = pypsa.Network(counterfactual_path)
        start = pd.Timestamp(run["start_utc"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        horizon = int(run["horizon_hours"])
        snapshots = pd.DatetimeIndex(baseline.snapshots)
        if snapshots.tz is None:
            snapshots = snapshots.tz_localize("UTC")
        else:
            snapshots = snapshots.tz_convert("UTC")
        selected = snapshots[
            (snapshots >= start)
            & (snapshots < start + pd.Timedelta(hours=horizon))
        ]
        if selected.empty:
            raise ValueError(
                f"PyPSA-Eur run {run['run_id']} has no requested snapshots"
            )
        baseline_dispatch = baseline.generators_t.p.copy()
        counterfactual_dispatch = counterfactual.generators_t.p.copy()
        baseline_dispatch.index = snapshots
        counterfactual_snapshots = pd.DatetimeIndex(counterfactual.snapshots)
        if counterfactual_snapshots.tz is None:
            counterfactual_snapshots = counterfactual_snapshots.tz_localize("UTC")
        else:
            counterfactual_snapshots = counterfactual_snapshots.tz_convert("UTC")
        counterfactual_dispatch.index = counterfactual_snapshots
        weights = baseline.snapshot_weightings["generators"].copy()
        weights.index = snapshots
        delta = dispatch_emissions_delta(
            baseline_dispatch.loc[selected],
            counterfactual_dispatch.loc[selected],
            baseline.generators,
            weights.loc[selected],
            emission_factors,
            config["pypsa_carrier_to_emissions_source"],
        )
        perturbation = float(run["perturbation_mwh"])
        if perturbation <= 0:
            raise ValueError("PyPSA-Eur perturbation_mwh must be positive")
        records.append(
            {
                "run_id": run["run_id"],
                "start_utc": start,
                "horizon_hours": horizon,
                "structural_marginal_response_kgco2e_per_mwh": (
                    delta["emissions_delta_kgco2e"] / perturbation
                ),
                "perturbation_mwh": perturbation,
                "dispatch_mapping_coverage": delta["dispatch_mapping_coverage"],
                "absolute_dispatch_delta_mwh": delta["absolute_dispatch_delta_mwh"],
                "unmapped_generator_count": delta["unmapped_generator_count"],
                "structural_source": "pypsa_eur_paired_dispatch",
            }
        )
    if not records:
        raise ValueError("PyPSA-Eur counterfactual manifest contains no runs")
    return pd.DataFrame(records)


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def build_synthetic_dispatch_counterfactual(
    dataset: pd.DataFrame,
    config: dict[str, Any],
    emission_factors: dict[str, float],
) -> pd.DataFrame:
    required = {
        "decision_id",
        "state_timestamp_utc",
        "horizon_hours",
        "pre_load_forecast_mw",
        "pre_renewable_forecast_mw",
        "pre_planned_unavailable_capacity_mw",
        "pre_net_scheduled_import_mw",
        "pre_day_ahead_import_capacity_mw",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError("synthetic dispatch inputs missing: " + ", ".join(missing))
    states = dataset.copy()
    if "eligible_for_fit" in states:
        states = states[states["eligible_for_fit"]].copy()
    states = states.drop_duplicates(
        ["decision_id", "state_timestamp_utc", "horizon_hours"]
    )
    residual = (
        pd.to_numeric(states["pre_load_forecast_mw"], errors="coerce")
        - pd.to_numeric(states["pre_renewable_forecast_mw"], errors="coerce")
        + pd.to_numeric(
            states["pre_planned_unavailable_capacity_mw"], errors="coerce"
        ).fillna(0.0)
    )
    import_capacity = pd.to_numeric(
        states["pre_day_ahead_import_capacity_mw"], errors="coerce"
    ).clip(lower=1.0)
    net_import = pd.to_numeric(
        states["pre_net_scheduled_import_mw"], errors="coerce"
    ).fillna(0.0)
    import_utilization = (net_import.clip(lower=0) / import_capacity).clip(0, 1)
    low, high = residual.quantile([0.2, 0.8])
    local_factor = np.select(
        [residual <= low, residual >= high],
        [emission_factors["hydro"], emission_factors["coal"]],
        default=emission_factors["gas"],
    )
    neighbor_factor = np.select(
        [residual <= low, residual >= high],
        [emission_factors["nuclear"], emission_factors["gas"]],
        default=emission_factors["bioenergy"],
    )
    interconnection_share = (0.35 * (1 - import_utilization)).clip(0.05, 0.35)
    marginal_factor = (
        (1 - interconnection_share) * local_factor
        + interconnection_share * neighbor_factor
    )
    multiplier = states["horizon_hours"].astype(str).map(
        config["synthetic_storage_response_multiplier"]
    )
    states["structural_marginal_response_kgco2e_per_mwh"] = (
        marginal_factor * multiplier.fillna(1.0)
    )
    states["run_id"] = (
        states["decision_id"].astype(str)
        + ":"
        + states["state_timestamp_utc"].astype(str)
        + ":"
        + states["horizon_hours"].astype(str)
    )
    states["perturbation_mwh"] = 1.0
    states["dispatch_mapping_coverage"] = 1.0
    states["absolute_dispatch_delta_mwh"] = 1.0
    states["unmapped_generator_count"] = 0
    states["structural_source"] = "synthetic_zonal_merit_order_dispatch"
    return states[
        [
            "run_id",
            "state_timestamp_utc",
            "horizon_hours",
            "structural_marginal_response_kgco2e_per_mwh",
            "perturbation_mwh",
            "dispatch_mapping_coverage",
            "absolute_dispatch_delta_mwh",
            "unmapped_generator_count",
            "structural_source",
        ]
    ].rename(columns={"state_timestamp_utc": "start_utc"})


def load_statistical_estimates(
    constrained_path: str | Path,
    dml_path: str | Path,
) -> pd.DataFrame:
    constrained = pd.read_csv(constrained_path)
    constrained_rows = pd.DataFrame(
        {
            "horizon_hours": constrained["horizon_hours"].astype(int),
            "estimator": "constrained_regression",
            "statistical_estimate_kgco2e_per_mwh": constrained[
                "mean_marginal_response_kgco2e_per_mwh"
            ],
            "ci95_lower_kgco2e_per_mwh": constrained.get(
                "ci95_mean_marginal_response_lower_kgco2e_per_mwh"
            ),
            "ci95_upper_kgco2e_per_mwh": constrained.get(
                "ci95_mean_marginal_response_upper_kgco2e_per_mwh"
            ),
        }
    )
    dml = pd.read_csv(dml_path)
    dml_rows = pd.DataFrame(
        {
            "horizon_hours": dml["horizon_hours"].astype(int),
            "estimator": "double_machine_learning",
            "statistical_estimate_kgco2e_per_mwh": dml[
                "orthogonal_mean_marginal_response_kgco2e_per_mwh"
            ],
            "ci95_lower_kgco2e_per_mwh": np.nan,
            "ci95_upper_kgco2e_per_mwh": np.nan,
        }
    )
    return pd.concat([constrained_rows, dml_rows], ignore_index=True)


def compare_structural_and_statistical(
    structural: pd.DataFrame,
    statistical: pd.DataFrame,
    config: dict[str, Any],
    *,
    data_origin: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    structural_summary = structural.groupby("horizon_hours", as_index=False).agg(
        structural_estimate_kgco2e_per_mwh=(
            "structural_marginal_response_kgco2e_per_mwh",
            "mean",
        ),
        structural_run_count=("run_id", "nunique"),
        minimum_dispatch_mapping_coverage=("dispatch_mapping_coverage", "min"),
    )
    comparison = statistical.merge(
        structural_summary, on="horizon_hours", how="inner", validate="many_to_one"
    )
    floor = float(config["minimum_comparison_floor_kgco2e_per_mwh"])
    denominator = comparison["structural_estimate_kgco2e_per_mwh"].abs().clip(
        lower=floor
    )
    comparison["absolute_difference_kgco2e_per_mwh"] = (
        comparison["statistical_estimate_kgco2e_per_mwh"]
        - comparison["structural_estimate_kgco2e_per_mwh"]
    ).abs()
    comparison["relative_difference"] = (
        comparison["absolute_difference_kgco2e_per_mwh"] / denominator
    )
    comparison["sign_agreement"] = np.sign(
        comparison["statistical_estimate_kgco2e_per_mwh"]
    ).eq(np.sign(comparison["structural_estimate_kgco2e_per_mwh"]))
    comparison["structural_inside_statistical_ci95"] = (
        comparison["ci95_lower_kgco2e_per_mwh"].notna()
        & comparison["structural_estimate_kgco2e_per_mwh"].between(
            comparison["ci95_lower_kgco2e_per_mwh"],
            comparison["ci95_upper_kgco2e_per_mwh"],
        )
    )
    coverage_ok = comparison["minimum_dispatch_mapping_coverage"].ge(
        float(config["minimum_dispatch_coverage"])
    )
    comparison["agreement"] = (
        comparison["sign_agreement"]
        & comparison["relative_difference"].le(
            float(config["maximum_relative_difference"])
        )
        & coverage_ok
    )
    comparison["model_risk"] = np.select(
        [~comparison["sign_agreement"], ~comparison["agreement"]],
        ["high", "elevated"],
        default="low",
    )
    if comparison.empty:
        status = "blocked"
        risk = "unknown"
        blockers = ["no_common_structural_and_statistical_horizons"]
    else:
        risk_order = {"low": 0, "elevated": 1, "high": 2}
        risk = max(comparison["model_risk"], key=risk_order.get)
        status = "agreement" if risk == "low" else "model_risk"
        blockers = []
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "validation_version": config["validation_version"],
        "status": status,
        "structural_validation_ready": not blockers,
        "data_origin": data_origin,
        "model_risk": risk,
        "blockers": blockers,
        "causal_claim_eligible": data_origin == "pypsa_eur" and risk == "low",
        "comparison_policy": (
            "Disagreement is retained as model-risk evidence; estimates are not "
            "averaged."
        ),
        "recommended_action": (
            "eligible_for_parallel_validation"
            if risk == "low"
            else "do_not_treat_as_structurally_validated"
        ),
        "thresholds": {
            "maximum_relative_difference": float(
                config["maximum_relative_difference"]
            ),
            "minimum_dispatch_coverage": float(config["minimum_dispatch_coverage"]),
        },
        "comparisons": comparison.astype(object)
        .where(pd.notna(comparison), None)
        .to_dict(orient="records"),
    }
    return report, comparison


def choose_statistical_artifacts() -> tuple[Path | None, Path | None, str]:
    observed_ready = (
        DEFAULT_OBSERVED_CONSTRAINED_PATH.exists()
        and DEFAULT_OBSERVED_DML_PATH.exists()
        and DEFAULT_OBSERVED_CONSTRAINED_PATH.stat().st_size > 1
        and DEFAULT_OBSERVED_DML_PATH.stat().st_size > 1
    )
    if observed_ready:
        return (
            DEFAULT_OBSERVED_CONSTRAINED_PATH,
            DEFAULT_OBSERVED_DML_PATH,
            "observed",
        )
    if (
        DEFAULT_SYNTHETIC_CONSTRAINED_PATH.exists()
        and DEFAULT_SYNTHETIC_DML_PATH.exists()
    ):
        return (
            DEFAULT_SYNTHETIC_CONSTRAINED_PATH,
            DEFAULT_SYNTHETIC_DML_PATH,
            "synthetic",
        )
    return None, None, "missing"


def write_structural_validation(
    report: dict[str, Any],
    comparison: pd.DataFrame,
    structural: pd.DataFrame,
) -> None:
    for path in (
        DEFAULT_REPORT_PATH,
        DEFAULT_COMPARISON_PATH,
        DEFAULT_STRUCTURAL_ESTIMATES_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    comparison.to_csv(DEFAULT_COMPARISON_PATH, index=False)
    structural.to_csv(DEFAULT_STRUCTURAL_ESTIMATES_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare statistical causal estimates with dispatch counterfactuals."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--pypsa-manifest", default=str(DEFAULT_PYPSA_MANIFEST_PATH))
    parser.add_argument(
        "--external-estimates", default=str(DEFAULT_EXTERNAL_ESTIMATES_PATH)
    )
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    config = load_validation_config(args.config)
    factors = load_emission_factor_config(DEFAULT_FACTORS_PATH)[
        "direct_operational_emissions"
    ]
    constrained_path, dml_path, statistical_origin = choose_statistical_artifacts()
    if constrained_path is None or dml_path is None:
        report = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "validation_version": config["validation_version"],
            "status": "blocked",
            "structural_validation_ready": False,
            "data_origin": "missing",
            "model_risk": "unknown",
            "blockers": ["missing_statistical_estimator_artifacts"],
            "causal_claim_eligible": False,
        }
        comparison = pd.DataFrame()
        structural = pd.DataFrame()
    else:
        if Path(args.pypsa_manifest).exists():
            structural = load_pypsa_eur_counterfactuals(
                args.pypsa_manifest, config, factors
            )
            structural_origin = "pypsa_eur"
        elif Path(args.external_estimates).exists():
            structural = pd.read_csv(args.external_estimates)
            structural_origin = "external_dispatch"
        elif (
            bool(config["allow_synthetic_dispatch_fallback"])
            and DEFAULT_SYNTHETIC_DATASET_PATH.exists()
        ):
            structural = build_synthetic_dispatch_counterfactual(
                pd.read_parquet(DEFAULT_SYNTHETIC_DATASET_PATH), config, factors
            )
            structural_origin = "synthetic_dispatch"
        else:
            structural = pd.DataFrame()
            structural_origin = "missing"
        if structural.empty:
            report = {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "validation_version": config["validation_version"],
                "status": "blocked",
                "structural_validation_ready": False,
                "data_origin": structural_origin,
                "model_risk": "unknown",
                "blockers": ["missing_dispatch_counterfactuals"],
                "causal_claim_eligible": False,
            }
            comparison = pd.DataFrame()
        else:
            statistical = load_statistical_estimates(constrained_path, dml_path)
            report, comparison = compare_structural_and_statistical(
                structural,
                statistical,
                config,
                data_origin=structural_origin,
            )
            report["statistical_data_origin"] = statistical_origin
            if structural_origin == "synthetic_dispatch":
                report["product_disclosure"] = (
                    "Structural response is generated by the synthetic zonal dispatch "
                    "fallback, not a solved PyPSA-Eur network."
                )
    write_structural_validation(report, comparison, structural)
    print(
        json.dumps(
            {
                "status": report["status"],
                "structural_validation_ready": report[
                    "structural_validation_ready"
                ],
                "data_origin": report["data_origin"],
                "model_risk": report["model_risk"],
                "causal_claim_eligible": report["causal_claim_eligible"],
                "blockers": report["blockers"],
                "output": str(DEFAULT_REPORT_PATH),
            },
            indent=2,
        )
    )
    return 2 if args.require_ready and not report["structural_validation_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
