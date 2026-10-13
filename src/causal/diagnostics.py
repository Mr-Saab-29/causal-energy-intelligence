"""Causal identification diagnostics for observed workload decisions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.causal.dml import (
    OUTCOME,
    TREATMENT,
    build_effect_modifiers,
    feature_medians,
    load_dml_config,
    nuisance_model,
    numeric_features,
)
from src.causal.marginal_response import fit_propensity, time_block_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/causal_diagnostics.json"
DEFAULT_DATASET_PATH = (
    ROOT / "reports/causal/observed_treatment_estimator_dataset.parquet"
)
DEFAULT_MODEL_PATH = ROOT / "models/causal/time_blocked_dml.joblib"
DEFAULT_SYNTHETIC_DATASET_PATH = (
    ROOT / "reports/demo/causal/observed_treatment_estimator_dataset.parquet"
)
DEFAULT_SYNTHETIC_MODEL_PATH = ROOT / "models/demo/causal/time_blocked_dml.joblib"
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_diagnostics.json"
DEFAULT_DETAILS_PATH = ROOT / "reports/causal/causal_diagnostic_details.csv"


def load_diagnostics_config(
    diagnostics_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_dml_config()
    diagnostics = json.loads(Path(diagnostics_path).read_text(encoding="utf-8"))
    required = {
        "diagnostics_version",
        "maximum_weighted_absolute_smd",
        "minimum_negative_control_p_value",
        "minimum_placebo_p_value",
        "minimum_robustness_value",
        "minimum_regime_decisions",
        "allow_synthetic_product_fallback",
        "generate_synthetic_fallback_if_missing",
        "bootstrap_iterations",
        "bootstrap_block_days",
        "negative_control_outcomes",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError("causal diagnostics config missing: " + ", ".join(missing))
    return {**config, **diagnostics}


def add_placebo_treatments(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(
        ["decision_at_utc", "decision_id", "state_timestamp_utc"]
    ).copy()
    output["state_ordinal"] = output.groupby("decision_id").cumcount()
    output["placebo_shifted_treatment_mwh"] = output.groupby(
        "decision_id", observed=True
    )[TREATMENT].transform(lambda values: np.roll(values.to_numpy(), 1))
    output["lead_treatment_mwh"] = output.groupby(
        "state_ordinal", observed=True
    )[TREATMENT].shift(-1)
    return output


def standardized_mean_difference(
    values: np.ndarray,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
) -> float | None:
    valid = np.isfinite(values)
    values = values[valid]
    treated = treated[valid]
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = weights[valid]
    if not treated.any() or treated.all():
        return None

    def moments(mask: np.ndarray) -> tuple[float, float]:
        group_values = values[mask]
        group_weights = weights[mask]
        mean = float(np.average(group_values, weights=group_weights))
        variance = float(
            np.average((group_values - mean) ** 2, weights=group_weights)
        )
        return mean, variance

    treated_mean, treated_variance = moments(treated)
    control_mean, control_variance = moments(~treated)
    pooled = math.sqrt(max((treated_variance + control_variance) / 2, 0.0))
    if pooled < 1e-12:
        return 0.0
    return (treated_mean - control_mean) / pooled


def assess_overlap_and_balance(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    propensity, _, overlap = fit_propensity(train, test, config)
    columns = list(config["adjustment_features"])
    medians = train[columns].median(numeric_only=True).reindex(columns).fillna(0.0)
    x_test = test[columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
    probability = np.clip(propensity.predict_proba(x_test)[:, 1], 0.02, 0.98)
    treated = test["treatment_observed"].astype(bool).to_numpy()
    weights = np.where(treated, 1 / probability, 1 / (1 - probability))
    weights = np.minimum(weights, np.quantile(weights, 0.99))
    lower = float(config["propensity_lower_bound"])
    upper = float(config["propensity_upper_bound"])
    in_bounds = (probability >= lower) & (probability <= upper)
    overlap.update(
        {
            "status": (
                "pass"
                if float(in_bounds.mean()) >= float(config["minimum_overlap_share"])
                else "fail"
            ),
            "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
            "treated_rows": int(treated.sum()),
            "control_rows": int((~treated).sum()),
        }
    )
    details = []
    for column in columns:
        values = pd.to_numeric(test[column], errors="coerce").to_numpy(float)
        raw = standardized_mean_difference(values, treated)
        weighted = standardized_mean_difference(values, treated, weights)
        details.append(
            {
                "horizon_hours": horizon,
                "diagnostic": "covariate_balance",
                "dimension": column,
                "estimate": weighted,
                "raw_estimate": raw,
                "rows": int(np.isfinite(values).sum()),
            }
        )
    absolute = [abs(row["estimate"]) for row in details if row["estimate"] is not None]
    maximum = max(absolute, default=0.0)
    threshold = float(config["maximum_weighted_absolute_smd"])
    balance = {
        "status": "pass" if maximum <= threshold else "fail",
        "maximum_absolute_weighted_smd": maximum,
        "mean_absolute_weighted_smd": float(np.mean(absolute)) if absolute else 0.0,
        "features_above_threshold": sum(value > threshold for value in absolute),
        "threshold": threshold,
    }
    return overlap, balance, details


def out_of_sample_residuals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    features = [
        column
        for column in config["adjustment_features"]
        if column != outcome_column
    ]
    medians = feature_medians(train, features)
    x_train = numeric_features(train, features, medians)
    x_test = numeric_features(test, features, medians)
    train_treatment = pd.to_numeric(train[treatment_column], errors="coerce")
    test_treatment = pd.to_numeric(test[treatment_column], errors="coerce")
    train_outcome = pd.to_numeric(train[outcome_column], errors="coerce")
    test_outcome = pd.to_numeric(test[outcome_column], errors="coerce")
    valid_train = train_treatment.notna() & train_outcome.notna()
    valid_test = test_treatment.notna() & test_outcome.notna()
    if valid_train.sum() < 20 or valid_test.sum() < 10:
        return pd.DataFrame()
    outcome_model = nuisance_model(config, 710).fit(
        x_train[valid_train], train_outcome[valid_train]
    )
    treatment_model = nuisance_model(config, 711).fit(
        x_train[valid_train], train_treatment[valid_train]
    )
    result = test.loc[valid_test, ["decision_id", "decision_at_utc"]].copy()
    result["outcome_residual"] = (
        test_outcome[valid_test].to_numpy()
        - outcome_model.predict(x_test[valid_test])
    )
    result["treatment_residual"] = (
        test_treatment[valid_test].to_numpy()
        - treatment_model.predict(x_test[valid_test])
    )
    return result


def block_bootstrap_association(
    residuals: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    if residuals.empty:
        return {"status": "not_available", "reason": "insufficient_complete_rows"}
    treatment = residuals["treatment_residual"].to_numpy(float)
    outcome = residuals["outcome_residual"].to_numpy(float)
    denominator = float(np.sum(treatment**2))
    if denominator < 1e-12:
        return {"status": "not_available", "reason": "no_residual_treatment_variation"}
    estimate = float(np.sum(treatment * outcome) / denominator)
    rows = residuals.copy()
    decision_date = pd.to_datetime(rows["decision_at_utc"], utc=True).dt.floor("D")
    origin = decision_date.min()
    block_days = int(config["bootstrap_block_days"])
    rows["block"] = ((decision_date - origin).dt.days // block_days).astype(int)
    blocks = rows["block"].unique()
    rng = np.random.default_rng(int(config["random_seed"]) + seed_offset)
    draws = []
    for _ in range(int(config["bootstrap_iterations"])):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        parts = [rows[rows["block"].eq(block)] for block in sampled]
        sample = pd.concat(parts, ignore_index=True)
        sample_treatment = sample["treatment_residual"].to_numpy(float)
        sample_denominator = float(np.sum(sample_treatment**2))
        if sample_denominator > 1e-12:
            draws.append(
                float(
                    np.sum(
                        sample_treatment
                        * sample["outcome_residual"].to_numpy(float)
                    )
                    / sample_denominator
                )
            )
    if not draws:
        return {"status": "not_available", "reason": "bootstrap_failed"}
    draw_array = np.asarray(draws)
    intervals = {}
    for level in config["confidence_levels"]:
        alpha = (1 - float(level)) / 2
        intervals[f"ci{int(round(float(level) * 100))}"] = {
            "lower": float(np.quantile(draw_array, alpha)),
            "upper": float(np.quantile(draw_array, 1 - alpha)),
        }
    nonpositive = (int(np.sum(draw_array <= 0)) + 1) / (len(draw_array) + 1)
    nonnegative = (int(np.sum(draw_array >= 0)) + 1) / (len(draw_array) + 1)
    p_value = min(1.0, 2 * min(nonpositive, nonnegative))
    return {
        "status": "available",
        "estimate": estimate,
        "block_days": block_days,
        "bootstrap_successful_iterations": len(draws),
        "bootstrap_standard_error": float(draw_array.std(ddof=1)),
        "two_sided_p_value": p_value,
        "intervals": intervals,
    }


def robustness_value(association: dict[str, Any], decisions: int) -> dict[str, Any]:
    standard_error = association.get("bootstrap_standard_error")
    if not standard_error or decisions < 3:
        return {"status": "not_available"}
    t_value = abs(float(association["estimate"]) / float(standard_error))
    degrees_of_freedom = decisions - 1
    partial_r2 = t_value**2 / (t_value**2 + degrees_of_freedom)
    f_value = t_value / math.sqrt(degrees_of_freedom)
    value = 0.5 * (math.sqrt(f_value**4 + 4 * f_value**2) - f_value**2)
    return {
        "status": "available",
        "method": (
            "Cinelli-Hazlett robustness-value approximation using block-bootstrap SE"
        ),
        "partial_r2_treatment_outcome": partial_r2,
        "robustness_value_to_zero": value,
        "interpretation": (
            "An omitted confounder would need this partial-R2 strength with both "
            "residual treatment and residual outcome to reduce the estimate to zero."
        ),
    }


def stability_diagnostics(
    test: pd.DataFrame,
    model: Any,
    config: dict[str, Any],
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modifiers = build_effect_modifiers(test, model.renewable_edges)
    rows = test[["decision_id", "decision_at_utc"]].copy()
    rows["effect"] = model.marginal_response(test)
    rows = pd.concat([rows, modifiers], axis=1)
    details = []
    sparse = []
    expected_levels = {
        "effect_season": ("winter", "spring", "summer", "autumn"),
        "effect_renewable_regime": ("low", "medium", "high"),
        "effect_outage_regime": ("no_outage", "outage"),
        "effect_congestion_regime": ("low", "medium", "high"),
    }
    for dimension, levels in expected_levels.items():
        for level in levels:
            group = rows[rows[dimension].eq(level)]
            label = f"{dimension.removeprefix('effect_')}:{level}"
            if group.empty:
                sparse.append(label)
                details.append(
                    {
                        "horizon_hours": horizon,
                        "diagnostic": "regime_stability",
                        "dimension": dimension.removeprefix("effect_"),
                        "level": level,
                        "estimate": None,
                        "rows": 0,
                        "decisions": 0,
                    }
                )
                continue
            interval = block_bootstrap_mean(
                group, config, seed_offset=len(details) + horizon
            )
            decisions = int(group["decision_id"].nunique())
            if decisions < int(config["minimum_regime_decisions"]):
                sparse.append(label)
            details.append(
                {
                    "horizon_hours": horizon,
                    "diagnostic": "regime_stability",
                    "dimension": dimension.removeprefix("effect_"),
                    "level": str(level),
                    "estimate": float(group["effect"].mean()),
                    "rows": len(group),
                    "decisions": decisions,
                    **interval,
                }
            )
    return {
        "status": "pass" if not sparse else "warn",
        "minimum_regime_decisions": int(config["minimum_regime_decisions"]),
        "sparse_regimes": sparse,
        "interval_method": (
            "decision-day calendar-block bootstrap of fitted heterogeneous effects"
        ),
    }, details


def block_bootstrap_mean(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed_offset: int,
) -> dict[str, float]:
    decision_effect = frame.groupby("decision_id", observed=True)["effect"].mean()
    decision_time = frame.groupby("decision_id", observed=True)[
        "decision_at_utc"
    ].first()
    values = pd.DataFrame({"effect": decision_effect, "time": decision_time}).dropna()
    if values.empty:
        return {}
    dates = pd.to_datetime(values["time"], utc=True).dt.floor("D")
    values["block"] = (
        (dates - dates.min()).dt.days // int(config["bootstrap_block_days"])
    ).astype(int)
    blocks = values["block"].unique()
    rng = np.random.default_rng(int(config["random_seed"]) + seed_offset)
    draws = []
    for _ in range(int(config["bootstrap_iterations"])):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        draws.append(
            float(
                pd.concat([values[values["block"].eq(block)] for block in sampled])[
                    "effect"
                ].mean()
            )
        )
    result = {}
    for level in config["confidence_levels"]:
        alpha = (1 - float(level)) / 2
        label = int(round(float(level) * 100))
        result[f"ci{label}_lower"] = float(np.quantile(draws, alpha))
        result[f"ci{label}_upper"] = float(np.quantile(draws, 1 - alpha))
    return result


def diagnose_horizon(
    dataset: pd.DataFrame,
    artifact: dict[str, Any],
    horizon: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = dataset[
        dataset["horizon_hours"].eq(horizon) & dataset["eligible_for_fit"]
    ].copy()
    frame = add_placebo_treatments(frame)
    train, test, split = time_block_split(
        frame, float(config["train_share"]), int(config["embargo_hours"])
    )
    overlap, balance, details = assess_overlap_and_balance(
        train, test, config, horizon
    )
    checks: dict[str, Any] = {}
    tests = {
        "observed_treatment": (TREATMENT, OUTCOME),
        "future_treatment_lead": ("lead_treatment_mwh", OUTCOME),
        "within_decision_shift_placebo": ("placebo_shifted_treatment_mwh", OUTCOME),
    }
    for outcome in config["negative_control_outcomes"]:
        if outcome in frame.columns:
            tests[f"negative_control:{outcome}"] = (TREATMENT, outcome)
    for offset, (name, (treatment, outcome)) in enumerate(tests.items()):
        residuals = out_of_sample_residuals(
            train, test, treatment, outcome, config
        )
        checks[name] = block_bootstrap_association(
            residuals, config, seed_offset=1000 * horizon + offset
        )
        details.append(
            {
                "horizon_hours": horizon,
                "diagnostic": name,
                "dimension": outcome,
                "estimate": checks[name].get("estimate"),
                "p_value": checks[name].get("two_sided_p_value"),
                "rows": len(residuals),
            }
        )
    sensitivity = robustness_value(
        checks["observed_treatment"], int(test["decision_id"].nunique())
    )
    sensitivity["status"] = (
        "pass"
        if sensitivity.get("robustness_value_to_zero", 0)
        >= float(config["minimum_robustness_value"])
        else "fail"
    )
    stability, stability_details = stability_diagnostics(
        test, artifact["dml_model"], config, horizon
    )
    details.extend(stability_details)
    placebo_names = ["future_treatment_lead", "within_decision_shift_placebo"]
    placebo_pass = all(
        checks[name].get("two_sided_p_value", 0)
        >= float(config["minimum_placebo_p_value"])
        for name in placebo_names
    )
    negative_names = [name for name in checks if name.startswith("negative_control:")]
    negative_pass = bool(negative_names) and all(
        checks[name].get("two_sided_p_value", 0)
        >= float(config["minimum_negative_control_p_value"])
        for name in negative_names
    )
    statuses = [
        overlap["status"],
        balance["status"],
        "pass" if placebo_pass else "fail",
        "pass" if negative_pass else "fail",
        sensitivity["status"],
    ]
    return {
        "horizon_hours": horizon,
        "status": "pass" if all(value == "pass" for value in statuses) else "warn",
        "split": split,
        "treatment_overlap_and_positivity": overlap,
        "covariate_balance": balance,
        "lead_and_placebo_tests": {
            "status": "pass" if placebo_pass else "fail",
            **{name: checks[name] for name in placebo_names},
        },
        "negative_controls": {
            "status": "pass" if negative_pass else "fail",
            "tests": {
                name.removeprefix("negative_control:"): checks[name]
                for name in negative_names
            },
        },
        "unobserved_confounding_sensitivity": sensitivity,
        "season_and_grid_regime_stability": stability,
        "time_series_uncertainty": {
            "status": (
                "pass"
                if checks["observed_treatment"].get("status") == "available"
                else "fail"
            ),
            "association": checks["observed_treatment"],
            "method": (
                "held-out residual association with decision-day "
                "calendar-block bootstrap"
            ),
        },
    }, details


def run_causal_diagnostics(
    dataset: pd.DataFrame,
    models: dict[str, Any],
    config: dict[str, Any],
    *,
    data_origin: str = "observed",
) -> tuple[dict[str, Any], pd.DataFrame]:
    horizons = []
    details: list[dict[str, Any]] = []
    blockers = []
    for horizon in config["response_horizons_hours"]:
        artifact = models.get(str(horizon))
        if artifact is None:
            blockers.append(f"missing_dml_model:{horizon}h")
            continue
        report, horizon_details = diagnose_horizon(
            dataset, artifact, int(horizon), config
        )
        horizons.append(report)
        details.extend(horizon_details)
    if blockers:
        status = "blocked"
    elif all(row["status"] == "pass" for row in horizons):
        status = "ok"
    else:
        status = "warn"
    synthetic_product = data_origin == "synthetic_product_bootstrap"
    causal_claim_eligible = data_origin == "observed" and status == "ok"
    product_eligible = causal_claim_eligible or (
        synthetic_product and status in {"ok", "warn"}
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "diagnostics_version": config["diagnostics_version"],
        "status": status,
        "diagnostics_ready": not blockers,
        "data_origin": data_origin,
        "production_eligible": product_eligible,
        "causal_claim_eligible": causal_claim_eligible,
        "evidence_tier": (
            "synthetic_product_bootstrap"
            if synthetic_product
            else "observational_diagnostics"
        ),
        "blockers": blockers,
        "horizons": horizons,
        "external_requirements": {
            "new_paid_data_or_api_required": False,
            "real_world_causal_claims_require_observed_completed_workloads": True,
        },
    }
    return report, pd.DataFrame(details)


def write_diagnostics(
    report: dict[str, Any],
    details: pd.DataFrame,
    *,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    details_path: str | Path = DEFAULT_DETAILS_PATH,
) -> None:
    report_path = Path(report_path)
    details_path = Path(details_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    details.to_csv(details_path, index=False)


def blocked_report(reason: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "diagnostics_version": config["diagnostics_version"],
        "status": "blocked",
        "diagnostics_ready": False,
        "data_origin": "observed",
        "production_eligible": False,
        "causal_claim_eligible": False,
        "evidence_tier": "not_available",
        "blockers": [reason],
        "horizons": [],
        "external_requirements": {
            "new_paid_data_or_api_required": False,
            "real_world_causal_claims_require_observed_completed_workloads": True,
        },
    }


def select_diagnostic_inputs(
    dataset_path: str | Path,
    model_path: str | Path,
    config: dict[str, Any],
) -> tuple[Path | None, Path | None, str, str | None, str | None]:
    dataset = Path(dataset_path)
    models = Path(model_path)
    if dataset.exists() and models.exists():
        return dataset, models, "observed", None, None
    if bool(config["allow_synthetic_product_fallback"]):
        synthetic_dataset = DEFAULT_SYNTHETIC_DATASET_PATH
        synthetic_models = DEFAULT_SYNTHETIC_MODEL_PATH
        if synthetic_dataset.exists() and synthetic_models.exists():
            reason = (
                "observed DML artifacts were unavailable; using the approved "
                "synthetic product bootstrap"
            )
            return (
                synthetic_dataset,
                synthetic_models,
                "synthetic_product_bootstrap",
                None,
                reason,
            )
    blocker = (
        "missing_observed_treatment_estimator_dataset"
        if not dataset.exists()
        else "missing_ready_dml_models"
    )
    return None, None, "observed", blocker, None


def ensure_synthetic_fallback(config: dict[str, Any]) -> None:
    if not bool(config["allow_synthetic_product_fallback"]):
        return
    if DEFAULT_SYNTHETIC_DATASET_PATH.exists() and DEFAULT_SYNTHETIC_MODEL_PATH.exists():
        return
    if not bool(config["generate_synthetic_fallback_if_missing"]):
        return
    from src.causal.synthetic_demo import run_synthetic_demo

    result = run_synthetic_demo()
    if result["status"] != "ok":
        raise RuntimeError("synthetic causal product bootstrap failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run causal identification diagnostics."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--models", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    config = load_diagnostics_config(args.config)
    if not (Path(args.dataset).exists() and Path(args.models).exists()):
        ensure_synthetic_fallback(config)
    dataset_path, model_path, data_origin, blocker, fallback_reason = (
        select_diagnostic_inputs(args.dataset, args.models, config)
    )
    if blocker is not None or dataset_path is None or model_path is None:
        report = blocked_report(blocker or "missing_diagnostic_inputs", config)
        details = pd.DataFrame()
    else:
        report, details = run_causal_diagnostics(
            pd.read_parquet(dataset_path),
            joblib.load(model_path),
            config,
            data_origin=data_origin,
        )
        report["input_dataset"] = str(dataset_path)
        report["input_models"] = str(model_path)
        if data_origin == "synthetic_product_bootstrap":
            report["fallback_reason"] = fallback_reason
            report["product_disclosure"] = (
                "Synthetic workload evidence is being used as operational product "
                "data; results are not real-user causal validation."
            )
    write_diagnostics(report, details)
    print(
        json.dumps(
            {
                "status": report["status"],
                "diagnostics_ready": report["diagnostics_ready"],
                "data_origin": report["data_origin"],
                "production_eligible": report["production_eligible"],
                "causal_claim_eligible": report["causal_claim_eligible"],
                "blockers": report["blockers"],
                "output": str(DEFAULT_REPORT_PATH),
            },
            indent=2,
        )
    )
    return 2 if args.require_ready and not report["diagnostics_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
