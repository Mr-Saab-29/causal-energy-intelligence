"""Time-blocked Double Machine Learning companion to the constrained estimator."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from src.causal.marginal_response import (
    ConstrainedMarginalResponseRegressor,
    build_estimator_dataset,
    decision_effects,
    evaluate_estimator_readiness,
    fit_propensity,
    load_estimator_config,
    time_block_split,
)
from src.causal.treatment_observations import load_treatment_observations
from src.data.load import create_database_engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DML_CONFIG_PATH = ROOT / "config/causal_dml.json"
DEFAULT_BASE_CONFIG_PATH = ROOT / "config/causal_estimator.json"
DEFAULT_MART_PATH = ROOT / "reports/causal/causal_hourly_feature_mart.parquet"
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_dml_estimator.json"
DEFAULT_ESTIMATES_PATH = ROOT / "reports/causal/dml_marginal_response_estimates.csv"
DEFAULT_HETEROGENEITY_PATH = ROOT / "reports/causal/dml_heterogeneous_effects.csv"
DEFAULT_COMPARISON_PATH = ROOT / "reports/causal/dml_estimator_comparison.csv"
DEFAULT_MODEL_PATH = ROOT / "models/causal/time_blocked_dml.joblib"

OUTCOME = "outcome_interconnected_emissions_kgco2e"
TREATMENT = "treatment_mwh"
MODIFIER_COLUMNS = (
    "effect_hour",
    "effect_season",
    "effect_renewable_regime",
    "effect_outage_regime",
    "effect_congestion_regime",
)
MODIFIER_CATEGORIES = [
    [f"h{hour:02d}" for hour in range(24)],
    ["winter", "spring", "summer", "autumn"],
    ["low", "medium", "high"],
    ["no_outage", "outage"],
    ["low", "medium", "high"],
]


@dataclass
class DMLMarginalResponseModel:
    """Orthogonal effect model with flexible fitted nuisance functions."""

    adjustment_features: list[str]
    encoder: OneHotEncoder
    effect_coefficients: np.ndarray
    renewable_edges: tuple[float, float]
    feature_medians: pd.Series
    outcome_model: HistGradientBoostingRegressor
    treatment_model: HistGradientBoostingRegressor

    def marginal_response(self, frame: pd.DataFrame) -> np.ndarray:
        modifiers = build_effect_modifiers(frame, self.renewable_edges)
        encoded = self.encoder.transform(modifiers[list(MODIFIER_COLUMNS)])
        design = np.column_stack([np.ones(len(frame)), encoded])
        return design @ self.effect_coefficients

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = numeric_features(frame, self.adjustment_features, self.feature_medians)
        treatment = (
            pd.to_numeric(frame[TREATMENT], errors="coerce").fillna(0.0).to_numpy()
        )
        outcome_nuisance = self.outcome_model.predict(x)
        treatment_nuisance = self.treatment_model.predict(x)
        return outcome_nuisance + self.marginal_response(frame) * (
            treatment - treatment_nuisance
        )


def load_dml_config(
    dml_path: str | Path = DEFAULT_DML_CONFIG_PATH,
    base_path: str | Path = DEFAULT_BASE_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_estimator_config(base_path)
    dml = json.loads(Path(dml_path).read_text(encoding="utf-8"))
    required = {
        "estimator_version",
        "cross_fit_folds",
        "cross_fit_initial_train_share",
        "nuisance_model",
        "effect_ridge_alpha",
    }
    missing = sorted(required - set(dml))
    if missing:
        raise ValueError("causal DML config missing: " + ", ".join(missing))
    return {**config, **dml, "base_estimator_version": config["estimator_version"]}


def nuisance_model(
    config: dict[str, Any], seed_offset: int = 0
) -> HistGradientBoostingRegressor:
    options = config["nuisance_model"]
    return HistGradientBoostingRegressor(
        max_iter=int(options["max_iter"]),
        learning_rate=float(options["learning_rate"]),
        max_leaf_nodes=int(options["max_leaf_nodes"]),
        min_samples_leaf=int(options["min_samples_leaf"]),
        l2_regularization=float(options["l2_regularization"]),
        random_state=int(config["random_seed"]) + seed_offset,
    )


def rolling_time_cross_fit(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Create nuisance residuals using only earlier, embargoed decision blocks."""
    decisions = (
        frame[["decision_id", "decision_at_utc"]]
        .drop_duplicates("decision_id")
        .sort_values("decision_at_utc")
        .reset_index(drop=True)
    )
    initial = max(
        int(config["minimum_cross_fit_train_decisions"]),
        int(math.ceil(len(decisions) * float(config["cross_fit_initial_train_share"]))),
    )
    if initial >= len(decisions):
        return pd.DataFrame(), []
    validation_blocks = np.array_split(
        np.arange(initial, len(decisions)), int(config["cross_fit_folds"])
    )
    outputs: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    features = list(config["adjustment_features"])
    embargo = pd.Timedelta(hours=int(config["embargo_hours"]))
    for fold, positions in enumerate(validation_blocks, start=1):
        if not len(positions):
            continue
        validation_decisions = decisions.iloc[positions]
        validation_start = pd.Timestamp(validation_decisions["decision_at_utc"].min())
        train_decisions = decisions[
            pd.to_datetime(decisions["decision_at_utc"], utc=True)
            < validation_start - embargo
        ]
        if len(train_decisions) < int(
            config["minimum_cross_fit_train_decisions"]
        ) or len(validation_decisions) < int(
            config["minimum_cross_fit_validation_decisions"]
        ):
            continue
        train = frame[frame["decision_id"].isin(train_decisions["decision_id"])].copy()
        validation = frame[
            frame["decision_id"].isin(validation_decisions["decision_id"])
        ].copy()
        medians = feature_medians(train, features)
        x_train = numeric_features(train, features, medians)
        x_validation = numeric_features(validation, features, medians)
        outcome = nuisance_model(config, fold * 2).fit(x_train, train[OUTCOME])
        treatment = nuisance_model(config, fold * 2 + 1).fit(x_train, train[TREATMENT])
        validation["outcome_nuisance_prediction"] = outcome.predict(x_validation)
        validation["treatment_nuisance_prediction"] = treatment.predict(x_validation)
        validation["outcome_residual"] = (
            validation[OUTCOME] - validation["outcome_nuisance_prediction"]
        )
        validation["treatment_residual"] = (
            validation[TREATMENT] - validation["treatment_nuisance_prediction"]
        )
        validation["cross_fit_fold"] = fold
        outputs.append(validation)
        metadata.append(
            {
                "fold": fold,
                "train_decisions": int(len(train_decisions)),
                "validation_decisions": int(len(validation_decisions)),
                "train_end_utc": pd.Timestamp(
                    train_decisions["decision_at_utc"].max()
                ).isoformat(),
                "validation_start_utc": validation_start.isoformat(),
            }
        )
    return (pd.concat(outputs).sort_index() if outputs else pd.DataFrame()), metadata


def fit_dml_horizon(
    dataset: pd.DataFrame,
    horizon: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    horizon_data = dataset[
        dataset["horizon_hours"].eq(horizon) & dataset["eligible_for_fit"]
    ].copy()
    train, test, split = time_block_split(
        horizon_data, float(config["train_share"]), int(config["embargo_hours"])
    )
    if split["train_decisions"] < int(config["minimum_train_decisions"]) or split[
        "test_decisions"
    ] < int(config["minimum_test_decisions"]):
        return blocked_horizon(horizon, split, "insufficient_train_or_test_decisions")

    cross_fit, folds = rolling_time_cross_fit(train, config)
    if len(folds) < int(config["cross_fit_folds"]):
        return blocked_horizon(
            horizon, split, "insufficient_time_cross_fit_folds", folds
        )
    treatment_residual_std = float(cross_fit["treatment_residual"].std())
    if treatment_residual_std < float(config["minimum_treatment_residual_std_mwh"]):
        return blocked_horizon(
            horizon, split, "weak_residual_treatment_variation", folds
        )

    renewable_edges = renewable_regime_edges(cross_fit)
    modifiers = build_effect_modifiers(cross_fit, renewable_edges)
    encoder = OneHotEncoder(
        categories=MODIFIER_CATEGORIES,
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
    encoded = encoder.fit_transform(modifiers[list(MODIFIER_COLUMNS)])
    effect_basis = np.column_stack([np.ones(len(cross_fit)), encoded])
    treatment_residual = cross_fit["treatment_residual"].to_numpy(float)
    design = treatment_residual[:, None] * effect_basis
    alpha = float(config["effect_ridge_alpha"])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ cross_fit["outcome_residual"].to_numpy(float),
    )

    features = list(config["adjustment_features"])
    medians = feature_medians(train, features)
    x_train = numeric_features(train, features, medians)
    model = DMLMarginalResponseModel(
        adjustment_features=features,
        encoder=encoder,
        effect_coefficients=coefficients,
        renewable_edges=renewable_edges,
        feature_medians=medians,
        outcome_model=nuisance_model(config, 100 + horizon).fit(
            x_train, train[OUTCOME]
        ),
        treatment_model=nuisance_model(config, 200 + horizon).fit(
            x_train, train[TREATMENT]
        ),
    )
    test_prediction = model.predict(test)
    test_actual = test[OUTCOME].to_numpy(float)
    metrics = {
        "rmse_kgco2e": float(mean_squared_error(test_actual, test_prediction) ** 0.5),
        "r2": float(r2_score(test_actual, test_prediction)),
    }
    decisions = decision_effects(test, model, horizon)
    heterogeneity = summarize_heterogeneity(test, model, horizon, renewable_edges)
    simple_model, comparison = compare_with_constrained(
        train, test, model, config, horizon
    )
    average, intervals = orthogonal_average_effect(cross_fit, config)
    report = {
        "horizon_hours": horizon,
        "status": "ok",
        "split": split,
        "cross_fit_folds": folds,
        "cross_fit_rows": int(len(cross_fit)),
        "treatment_residual_std_mwh": treatment_residual_std,
        "test_metrics": metrics,
        "point_estimates": {
            "orthogonal_mean_marginal_response_kgco2e_per_mwh": average,
            "heterogeneous_mean_marginal_response_kgco2e_per_mwh": float(
                model.marginal_response(test).mean()
            ),
        },
        "uncertainty_intervals": intervals,
        "comparison": comparison,
    }
    artifact = {
        "dml_model": model,
        "constrained_benchmark": simple_model,
        "report": report,
    }
    return report, decisions, heterogeneity, artifact


def blocked_horizon(
    horizon: int,
    split: dict[str, Any],
    blocker: str,
    folds: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, None]:
    return (
        {
            "horizon_hours": horizon,
            "status": "blocked",
            "blockers": [blocker],
            "split": split,
            "cross_fit_folds": folds or [],
        },
        pd.DataFrame(),
        pd.DataFrame(),
        None,
    )


def compare_with_constrained(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dml_model: DMLMarginalResponseModel,
    config: dict[str, Any],
    horizon: int,
) -> tuple[ConstrainedMarginalResponseRegressor, dict[str, Any]]:
    _, weights, _ = fit_propensity(train, test, config)
    simple = ConstrainedMarginalResponseRegressor(
        list(config["adjustment_features"]),
        list(config["marginal_state_features"]),
        float(config["ridge_alpha"]),
    ).fit(train, OUTCOME, sample_weight=weights)
    dml_effect = dml_model.marginal_response(test)
    simple_effect = simple.marginal_response(test)
    difference = float(abs(dml_effect.mean() - simple_effect.mean()))
    relative = difference / max(abs(float(simple_effect.mean())), 1e-9)
    correlation = safe_correlation(dml_effect, simple_effect)
    sign_disagreement = float(np.mean(np.sign(dml_effect) != np.sign(simple_effect)))
    agreement = relative <= float(
        config["comparison_max_relative_mean_difference"]
    ) and correlation >= float(config["comparison_min_effect_correlation"])
    return simple, {
        "horizon_hours": horizon,
        "status": "agreement" if agreement else "review",
        "recommended_action": (
            "retain_both_for_parallel_validation"
            if agreement
            else "retain_constrained_regression_and_investigate_dml_disagreement"
        ),
        "dml_mean_kgco2e_per_mwh": float(dml_effect.mean()),
        "constrained_mean_kgco2e_per_mwh": float(simple_effect.mean()),
        "absolute_mean_difference_kgco2e_per_mwh": difference,
        "relative_mean_difference": relative,
        "effect_correlation": correlation,
        "sign_disagreement_share": sign_disagreement,
    }


def orthogonal_average_effect(
    cross_fit: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[float, dict[str, dict[str, float]]]:
    treatment = cross_fit["treatment_residual"].to_numpy(float)
    outcome = cross_fit["outcome_residual"].to_numpy(float)
    denominator = float(np.mean(treatment**2))
    estimate = float(np.mean(treatment * outcome) / denominator)
    score = treatment * (outcome - estimate * treatment) / denominator
    cluster_score = (
        pd.Series(score).groupby(cross_fit["decision_id"].reset_index(drop=True)).sum()
    )
    standard_error = float(np.sqrt(np.sum(cluster_score.to_numpy() ** 2)) / len(score))
    intervals: dict[str, dict[str, float]] = {}
    for level in config["confidence_levels"]:
        critical = float(norm.ppf((1 + float(level)) / 2))
        label = f"ci{int(round(float(level) * 100))}"
        intervals[label] = {
            "lower_kgco2e_per_mwh": estimate - critical * standard_error,
            "upper_kgco2e_per_mwh": estimate + critical * standard_error,
            "cluster_robust_standard_error": standard_error,
        }
    return estimate, intervals


def renewable_regime_edges(frame: pd.DataFrame) -> tuple[float, float]:
    values = pd.to_numeric(frame["pre_renewable_forecast_mw"], errors="coerce")
    return float(values.quantile(1 / 3)), float(values.quantile(2 / 3))


def build_effect_modifiers(
    frame: pd.DataFrame,
    renewable_edges: tuple[float, float],
) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["state_timestamp_utc"], utc=True)
    renewable = pd.to_numeric(frame["pre_renewable_forecast_mw"], errors="coerce")
    renewable = renewable.fillna(float(np.mean(renewable_edges)))
    outage = pd.to_numeric(
        frame["pre_planned_unavailable_capacity_mw"], errors="coerce"
    ).fillna(0.0)
    net_import = pd.to_numeric(
        frame["pre_net_scheduled_import_mw"], errors="coerce"
    ).fillna(0.0)
    import_capacity = pd.to_numeric(
        frame["pre_day_ahead_import_capacity_mw"], errors="coerce"
    ).fillna(0.0)
    export_capacity = pd.to_numeric(
        frame["pre_day_ahead_export_capacity_mw"], errors="coerce"
    ).fillna(0.0)
    directional_capacity = np.where(net_import >= 0, import_capacity, export_capacity)
    utilization = np.divide(
        np.abs(net_import),
        directional_capacity,
        out=np.zeros(len(frame), dtype=float),
        where=directional_capacity > 0,
    )
    month = timestamp.dt.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )
    return pd.DataFrame(
        {
            "effect_hour": timestamp.dt.hour.map(lambda value: f"h{value:02d}"),
            "effect_season": season,
            "effect_renewable_regime": pd.cut(
                renewable,
                [-np.inf, renewable_edges[0], renewable_edges[1], np.inf],
                labels=["low", "medium", "high"],
                include_lowest=True,
            ).astype(str),
            "effect_outage_regime": np.where(outage > 0, "outage", "no_outage"),
            "effect_congestion_regime": pd.cut(
                utilization,
                [-np.inf, 0.5, 0.8, np.inf],
                labels=["low", "medium", "high"],
                include_lowest=True,
            ).astype(str),
        },
        index=frame.index,
    )


def summarize_heterogeneity(
    frame: pd.DataFrame,
    model: DMLMarginalResponseModel,
    horizon: int,
    renewable_edges: tuple[float, float],
) -> pd.DataFrame:
    modifiers = build_effect_modifiers(frame, renewable_edges)
    modifiers["effect_kgco2e_per_mwh"] = model.marginal_response(frame)
    records = []
    for dimension in MODIFIER_COLUMNS:
        summary = modifiers.groupby(dimension, observed=True)[
            "effect_kgco2e_per_mwh"
        ].agg(["mean", "count"])
        for level, row in summary.iterrows():
            records.append(
                {
                    "horizon_hours": horizon,
                    "dimension": dimension.removeprefix("effect_"),
                    "level": str(level),
                    "mean_marginal_response_kgco2e_per_mwh": float(row["mean"]),
                    "rows": int(row["count"]),
                }
            )
    return pd.DataFrame(records)


def feature_medians(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .median()
        .reindex(columns)
        .fillna(0.0)
    )


def numeric_features(
    frame: pd.DataFrame,
    columns: list[str],
    medians: pd.Series,
) -> np.ndarray:
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
        .to_numpy(float)
    )


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def run_dml_estimator(
    observations: pd.DataFrame,
    mart: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset = build_estimator_dataset(observations, mart, config)
    blockers = evaluate_estimator_readiness(observations, dataset, config)
    base = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "estimator_version": config["estimator_version"],
        "benchmark_estimator_version": config["base_estimator_version"],
        "uses_observed_workload_treatment": True,
        "uses_load_forecast_error_as_treatment": False,
        "completed_decisions": int(
            observations.get("status", pd.Series(dtype=str)).eq("completed").sum()
        ),
        "dataset_rows": int(len(dataset)),
    }
    if blockers:
        return (
            {
                **base,
                "status": "blocked",
                "dml_estimator_ready": False,
                "evidence_tier": "not_identified_readiness_blocked",
                "blockers": blockers,
                "horizons": [],
            },
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    horizon_reports = []
    decisions = []
    heterogeneity = []
    models: dict[str, Any] = {}
    for horizon in config["response_horizons_hours"]:
        report, horizon_decisions, horizon_heterogeneity, artifact = fit_dml_horizon(
            dataset, int(horizon), config
        )
        horizon_reports.append(report)
        if not horizon_decisions.empty:
            decisions.append(horizon_decisions)
        if not horizon_heterogeneity.empty:
            heterogeneity.append(horizon_heterogeneity)
        if artifact is not None:
            models[str(horizon)] = artifact
    horizon_blockers = [
        f"horizon_{row['horizon_hours']}h:{blocker}"
        for row in horizon_reports
        if row["status"] != "ok"
        for blocker in row.get("blockers", [])
    ]
    ready = not horizon_blockers
    return (
        {
            **base,
            "status": "ok" if ready else "blocked",
            "dml_estimator_ready": ready,
            "evidence_tier": (
                "observational_identified_under_stated_assumptions"
                if ready
                else "not_identified_readiness_blocked"
            ),
            "blockers": horizon_blockers,
            "horizons": horizon_reports,
        },
        pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(),
        pd.concat(heterogeneity, ignore_index=True)
        if heterogeneity
        else pd.DataFrame(),
        models,
    )


def write_dml_artifacts(
    report: dict[str, Any],
    decisions: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    models: dict[str, Any],
    *,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    estimates_path: str | Path = DEFAULT_ESTIMATES_PATH,
    heterogeneity_path: str | Path = DEFAULT_HETEROGENEITY_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    decision_effects_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    report_path = Path(report_path)
    estimates_path = Path(estimates_path)
    heterogeneity_path = Path(heterogeneity_path)
    comparison_path = Path(comparison_path)
    model_path = Path(model_path)
    decision_effects_path = (
        Path(decision_effects_path)
        if decision_effects_path is not None
        else estimates_path.with_name("dml_decision_effects.csv")
    )
    for path in (
        report_path,
        estimates_path,
        heterogeneity_path,
        comparison_path,
        decision_effects_path,
        model_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    estimates = pd.DataFrame(
        [
            {"horizon_hours": row["horizon_hours"], **row.get("point_estimates", {})}
            for row in report["horizons"]
            if row["status"] == "ok"
        ]
    )
    comparisons = pd.DataFrame(
        [row["comparison"] for row in report["horizons"] if row["status"] == "ok"]
    )
    estimates.to_csv(estimates_path, index=False)
    heterogeneity.to_csv(heterogeneity_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    decisions.to_csv(decision_effects_path, index=False)
    if report["dml_estimator_ready"]:
        joblib.dump(models, model_path)
    else:
        model_path.unlink(missing_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Fit time-blocked Double Machine Learning models."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--dml-config", default=str(DEFAULT_DML_CONFIG_PATH))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG_PATH))
    parser.add_argument("--mart-path", default=str(DEFAULT_MART_PATH))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")
    observations = load_treatment_observations(
        create_database_engine(args.database_url)
    )
    mart = pd.read_parquet(args.mart_path)
    config = load_dml_config(args.dml_config, args.base_config)
    report, decisions, heterogeneity, models = run_dml_estimator(
        observations, mart, config
    )
    write_dml_artifacts(report, decisions, heterogeneity, models)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dml_estimator_ready": report["dml_estimator_ready"],
                "completed_decisions": report["completed_decisions"],
                "blockers": report["blockers"],
                "output": str(DEFAULT_REPORT_PATH),
            },
            indent=2,
        )
    )
    return 2 if args.require_ready and not report["dml_estimator_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
