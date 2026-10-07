"""Observed-treatment marginal emissions estimator with strict identification gates."""

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
from scipy.optimize import lsq_linear
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.causal.treatment_observations import load_treatment_observations
from src.data.load import create_database_engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/causal_estimator.json"
DEFAULT_MART_PATH = ROOT / "reports/causal/causal_hourly_feature_mart.parquet"
DEFAULT_REPORT_PATH = ROOT / "reports/metrics/causal_identified_estimator.json"
DEFAULT_ESTIMATES_PATH = ROOT / "reports/causal/marginal_response_estimates.csv"
DEFAULT_DECISIONS_PATH = ROOT / "reports/causal/confidently_avoided_emissions.csv"
DEFAULT_DATASET_PATH = (
    ROOT / "reports/causal/observed_treatment_estimator_dataset.parquet"
)
DEFAULT_MODEL_PATH = ROOT / "models/causal/constrained_marginal_response.joblib"


@dataclass
class ConstrainedMarginalResponseRegressor:
    """Linear response surface whose treatment derivative is nonnegative."""

    adjustment_features: list[str]
    marginal_features: list[str]
    alpha: float = 1.0
    medians_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    marginal_min_: np.ndarray | None = None
    marginal_range_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None
    outcome_mean_: float = 0.0
    outcome_scale_: float = 1.0

    def fit(
        self,
        frame: pd.DataFrame,
        outcome_column: str,
        treatment_column: str = "treatment_mwh",
        sample_weight: np.ndarray | None = None,
    ) -> "ConstrainedMarginalResponseRegressor":
        x = (
            frame[self.adjustment_features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        self.medians_ = np.nanmedian(x, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        x = np.where(np.isfinite(x), x, self.medians_)
        self.means_ = x.mean(axis=0)
        self.scales_ = x.std(axis=0)
        self.scales_ = np.where(self.scales_ > 1e-9, self.scales_, 1.0)
        nuisance = (x - self.means_) / self.scales_

        marginal_indices = [
            self.adjustment_features.index(name) for name in self.marginal_features
        ]
        marginal_raw = x[:, marginal_indices]
        self.marginal_min_ = np.nanpercentile(marginal_raw, 1, axis=0)
        marginal_max = np.nanpercentile(marginal_raw, 99, axis=0)
        self.marginal_range_ = np.where(
            marginal_max - self.marginal_min_ > 1e-9,
            marginal_max - self.marginal_min_,
            1.0,
        )
        basis = self._basis_from_raw(marginal_raw)
        treatment = pd.to_numeric(frame[treatment_column], errors="coerce").to_numpy(
            float
        )
        design = np.column_stack(
            [np.ones(len(frame)), nuisance, treatment[:, None] * basis]
        )

        outcome = pd.to_numeric(frame[outcome_column], errors="coerce").to_numpy(float)
        self.outcome_mean_ = float(outcome.mean())
        self.outcome_scale_ = float(outcome.std()) or 1.0
        target = (outcome - self.outcome_mean_) / self.outcome_scale_
        weights = (
            np.ones(len(frame))
            if sample_weight is None
            else np.asarray(sample_weight, float)
        )
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)

        penalty = np.eye(design.shape[1]) * math.sqrt(max(self.alpha, 0.0))
        penalty[0, 0] = 0.0
        augmented_design = np.vstack([weighted_design, penalty])
        augmented_target = np.concatenate([weighted_target, np.zeros(design.shape[1])])
        marginal_start = 1 + len(self.adjustment_features)
        lower = np.full(design.shape[1], -np.inf)
        lower[marginal_start:] = 0.0
        result = lsq_linear(
            augmented_design,
            augmented_target,
            bounds=(lower, np.full(design.shape[1], np.inf)),
            lsmr_tol="auto",
        )
        if not result.success:
            raise RuntimeError(f"Constrained regression failed: {result.message}")
        self.coefficients_ = result.x
        return self

    def predict(
        self, frame: pd.DataFrame, treatment_mwh: np.ndarray | None = None
    ) -> np.ndarray:
        nuisance, basis = self._transform(frame)
        treatment = (
            pd.to_numeric(frame["treatment_mwh"], errors="coerce").to_numpy(float)
            if treatment_mwh is None
            else np.asarray(treatment_mwh, float)
        )
        design = np.column_stack(
            [np.ones(len(frame)), nuisance, treatment[:, None] * basis]
        )
        return self.outcome_mean_ + self.outcome_scale_ * (
            design @ self._coefficients()
        )

    def marginal_response(self, frame: pd.DataFrame) -> np.ndarray:
        _, basis = self._transform(frame)
        marginal_start = 1 + len(self.adjustment_features)
        return self.outcome_scale_ * (basis @ self._coefficients()[marginal_start:])

    def _transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if any(value is None for value in (self.medians_, self.means_, self.scales_)):
            raise RuntimeError("Estimator is not fitted")
        x = (
            frame[self.adjustment_features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        x = np.where(np.isfinite(x), x, self.medians_)
        nuisance = (x - self.means_) / self.scales_
        indices = [
            self.adjustment_features.index(name) for name in self.marginal_features
        ]
        return nuisance, self._basis_from_raw(x[:, indices])

    def _basis_from_raw(self, raw: np.ndarray) -> np.ndarray:
        scaled = np.clip((raw - self.marginal_min_) / self.marginal_range_, 0.0, 1.0)
        return np.column_stack([np.ones(len(raw)), scaled])

    def _coefficients(self) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("Estimator is not fitted")
        return self.coefficients_


def load_estimator_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "estimator_version",
        "response_horizons_hours",
        "adjustment_features",
        "marginal_state_features",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("causal estimator config missing: " + ", ".join(missing))
    if not set(config["marginal_state_features"]).issubset(
        config["adjustment_features"]
    ):
        raise ValueError("marginal_state_features must be adjustment_features")
    if any(
        not str(column).startswith("pre_") for column in config["adjustment_features"]
    ):
        raise ValueError("all estimator adjustment features must use the pre_ prefix")
    return config


def build_estimator_dataset(
    observations: pd.DataFrame,
    mart: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Expand completed decisions into treated and feasible alternative decision-hours."""
    columns = dataset_columns(config)
    if observations.empty or mart.empty:
        return pd.DataFrame(columns=columns)
    hourly = mart.copy()
    hourly["timestamp_utc"] = pd.to_datetime(
        hourly["timestamp_utc"], utc=True
    ).dt.floor("h")
    hourly = (
        hourly.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc")
        .set_index("timestamp_utc")
    )
    missing_features = sorted(set(config["adjustment_features"]) - set(hourly.columns))
    if missing_features:
        raise ValueError(
            "causal mart missing adjustment features: " + ", ".join(missing_features)
        )

    records: list[dict[str, Any]] = []
    for observation in observations.to_dict(orient="records"):
        if observation.get("status") != "completed":
            continue
        actual_start = utc_timestamp(observation.get("actual_start_utc"))
        actual_completion = utc_timestamp(observation.get("actual_completion_utc"))
        access_time = utc_timestamp(observation.get("dashboard_accessed_at_utc"))
        recommendation_time = utc_timestamp(
            observation.get("recommendation_generated_at_utc")
        )
        baseline = utc_timestamp(observation.get("baseline_start_utc"))
        recommended = utc_timestamp(observation.get("recommended_start_utc"))
        energy_kwh = numeric(observation.get("actual_energy_kwh"))
        if any(
            value is None
            for value in (
                actual_start,
                actual_completion,
                access_time,
                recommendation_time,
                baseline,
            )
        ):
            continue
        if energy_kwh is None or energy_kwh <= 0 or actual_completion <= actual_start:
            continue
        actual_hour = actual_start.floor("h")
        state_times: dict[pd.Timestamp, set[str]] = {}
        add_state_time(state_times, baseline, "baseline")
        add_state_time(state_times, actual_hour, "actual")
        add_state_time(state_times, recommended, "recommended")
        for candidate in parse_candidates(observation.get("candidate_alternatives")):
            add_state_time(
                state_times, utc_timestamp(candidate.get("timestamp_utc")), "candidate"
            )

        duration_hours = max(
            1, math.ceil((actual_completion - actual_start).total_seconds() / 3600)
        )
        for state_time, roles in sorted(state_times.items()):
            if state_time not in hourly.index:
                continue
            state = hourly.loc[state_time]
            snapshot_at = utc_timestamp(state.get("metadata_snapshot_at_utc"))
            for horizon in config["response_horizons_hours"]:
                response_end = state_time + pd.Timedelta(
                    hours=duration_hours + int(horizon)
                )
                outcome = forward_outcome_sum(hourly, state_time, response_end)
                if outcome is None:
                    continue
                treated = state_time == actual_hour
                overlap = intervals_overlap(
                    state_time,
                    response_end,
                    actual_start,
                    actual_completion,
                )
                record = {
                    "decision_id": str(observation["decision_id"]),
                    "decision_at_utc": access_time,
                    "state_timestamp_utc": state_time,
                    "state_roles": "|".join(sorted(roles)),
                    "horizon_hours": int(horizon),
                    "actual_energy_mwh": energy_kwh / 1000.0,
                    "treatment_mwh": energy_kwh / 1000.0 if treated else 0.0,
                    "treatment_observed": bool(treated),
                    "eligible_for_fit": bool(treated or not overlap),
                    "point_in_time_eligible": bool(
                        state.get("metadata_point_in_time_eligible", False)
                        and snapshot_at is not None
                        and snapshot_at <= recommendation_time
                    ),
                    "snapshot_at_utc": snapshot_at,
                    "information_cutoff_utc": recommendation_time,
                    "outcome_interconnected_emissions_kgco2e": outcome,
                    "workload_type": observation.get("workload_type"),
                    "baseline_source": observation.get("baseline_source"),
                }
                record.update(
                    {name: state.get(name) for name in config["adjustment_features"]}
                )
                records.append(record)
    return pd.DataFrame.from_records(records, columns=columns)


def evaluate_estimator_readiness(
    observations: pd.DataFrame,
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    completed = observations[
        observations.get("status", pd.Series(dtype=str)).eq("completed")
    ].copy()
    if len(completed) < int(config["minimum_completed_decisions"]):
        blockers.append(
            f"insufficient_completed_decisions:{len(completed)}:{config['minimum_completed_decisions']}"
        )
    if completed.empty:
        distinct_days = 0
    else:
        distinct_days = pd.to_datetime(
            completed["dashboard_accessed_at_utc"], utc=True, errors="coerce"
        ).dt.date.nunique()
    if distinct_days < int(config["minimum_distinct_decision_days"]):
        blockers.append(
            f"insufficient_decision_days:{distinct_days}:{config['minimum_distinct_decision_days']}"
        )
    if dataset.empty:
        blockers.append("no_observed_treatment_outcome_overlap")
        return blockers
    if not bool(dataset["point_in_time_eligible"].fillna(False).all()):
        blockers.append("pre_treatment_covariates_not_point_in_time_eligible")
    maximum_missing = float(config.get("maximum_adjustment_missing_share", 0.05))
    for feature in config["adjustment_features"]:
        missing_share = float(
            pd.to_numeric(dataset[feature], errors="coerce").isna().mean()
        )
        if missing_share > maximum_missing:
            blockers.append(
                f"adjustment_feature_missing:{feature}:{missing_share:.4f}:{maximum_missing:.4f}"
            )
    for horizon in config["response_horizons_hours"]:
        horizon_rows = dataset[
            dataset["horizon_hours"].eq(int(horizon)) & dataset["eligible_for_fit"]
        ]
        treated = horizon_rows[horizon_rows["treatment_observed"]]
        controls = horizon_rows[~horizon_rows["treatment_observed"]]
        if treated.empty:
            blockers.append(f"no_treated_rows:{horizon}h")
        if controls.empty:
            blockers.append(f"no_unexposed_feasible_alternatives:{horizon}h")
    return blockers


def time_block_split(
    frame: pd.DataFrame,
    train_share: float,
    embargo_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decisions = (
        frame[["decision_id", "decision_at_utc"]]
        .drop_duplicates("decision_id")
        .sort_values("decision_at_utc")
        .reset_index(drop=True)
    )
    split_index = max(1, min(len(decisions) - 1, int(len(decisions) * train_share)))
    train_decisions = decisions.iloc[:split_index]
    cutoff = pd.Timestamp(train_decisions["decision_at_utc"].max())
    test_start = cutoff + pd.Timedelta(hours=embargo_hours)
    test_decisions = decisions[decisions["decision_at_utc"] > test_start]
    train = frame[frame["decision_id"].isin(train_decisions["decision_id"])].copy()
    test = frame[frame["decision_id"].isin(test_decisions["decision_id"])].copy()
    metadata = {
        "train_end_utc": cutoff.isoformat(),
        "test_start_utc": test_start.isoformat(),
        "embargo_hours": int(embargo_hours),
        "train_decisions": int(train["decision_id"].nunique()),
        "test_decisions": int(test["decision_id"].nunique()),
    }
    return train, test, metadata


def fit_horizon(
    frame: pd.DataFrame,
    horizon: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    horizon_data = frame[frame["horizon_hours"].eq(int(horizon))].copy()
    data = horizon_data[horizon_data["eligible_for_fit"]].copy()
    train, test, split = time_block_split(
        data,
        float(config["train_share"]),
        int(config["embargo_hours"]),
    )
    blockers = []
    if split["train_decisions"] < int(config["minimum_train_decisions"]):
        blockers.append("insufficient_train_decisions")
    if split["test_decisions"] < int(config["minimum_test_decisions"]):
        blockers.append("insufficient_test_decisions")
    if blockers:
        return (
            {
                "horizon_hours": horizon,
                "status": "blocked",
                "blockers": blockers,
                "split": split,
            },
            pd.DataFrame(),
            None,
        )

    propensity, train_weights, overlap = fit_propensity(train, test, config)
    if overlap["overlap_share"] < float(config["minimum_overlap_share"]):
        return (
            {
                "horizon_hours": horizon,
                "status": "blocked",
                "blockers": ["insufficient_propensity_overlap"],
                "split": split,
                "overlap": overlap,
            },
            pd.DataFrame(),
            None,
        )

    model = ConstrainedMarginalResponseRegressor(
        list(config["adjustment_features"]),
        list(config["marginal_state_features"]),
        float(config["ridge_alpha"]),
    ).fit(train, "outcome_interconnected_emissions_kgco2e", sample_weight=train_weights)
    prediction = model.predict(test)
    actual = test["outcome_interconnected_emissions_kgco2e"].to_numpy(float)
    metrics = {
        "rmse_kgco2e": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
    }
    test_states = horizon_data[
        horizon_data["decision_id"].isin(test["decision_id"].unique())
    ].copy()
    decisions = decision_effects(test_states, model, horizon)
    bootstrap = bootstrap_effects(train, test_states, config, horizon)
    successful_bootstraps = int(
        bootstrap["bootstrap_iteration"].nunique() if not bootstrap.empty else 0
    )
    minimum_bootstraps = int(config["minimum_bootstrap_successful_iterations"])
    if successful_bootstraps < minimum_bootstraps:
        return (
            {
                "horizon_hours": horizon,
                "status": "blocked",
                "blockers": ["insufficient_successful_bootstrap_iterations"],
                "split": split,
                "overlap": overlap,
                "bootstrap_successful_iterations": successful_bootstraps,
                "minimum_bootstrap_successful_iterations": minimum_bootstraps,
            },
            pd.DataFrame(),
            None,
        )
    decisions = attach_intervals(decisions, bootstrap, config)
    estimates = summarize_effects(decisions, model, test, bootstrap, config, horizon)
    report = {
        "horizon_hours": horizon,
        "status": "ok",
        "split": split,
        "overlap": overlap,
        "test_metrics": metrics,
        **estimates,
    }
    artifact = {"model": model, "propensity_model": propensity, "report": report}
    return report, decisions, artifact


def fit_propensity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    columns = list(config["adjustment_features"])
    medians = train[columns].median(numeric_only=True).reindex(columns).fillna(0.0)
    x_train = train[columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
    x_test = test[columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y_train = train["treatment_observed"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("propensity model requires treated and control rows")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    ).fit(
        x_train,
        y_train,
    )
    train_probability = np.clip(model.predict_proba(x_train)[:, 1], 0.02, 0.98)
    test_probability = model.predict_proba(x_test)[:, 1]
    lower = float(config["propensity_lower_bound"])
    upper = float(config["propensity_upper_bound"])
    overlap = (test_probability >= lower) & (test_probability <= upper)
    weights = np.where(
        y_train.to_numpy() == 1, 1 / train_probability, 1 / (1 - train_probability)
    )
    weights = np.minimum(weights, np.quantile(weights, 0.99))
    return (
        model,
        weights,
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "overlap_share": round(float(overlap.mean()), 4),
            "test_propensity_min": round(float(test_probability.min()), 6),
            "test_propensity_max": round(float(test_probability.max()), 6),
        },
    )


def decision_effects(
    test: pd.DataFrame, model: ConstrainedMarginalResponseRegressor, horizon: int
) -> pd.DataFrame:
    records = []
    for decision_id, group in test.groupby("decision_id", observed=True):
        baseline = group[group["state_roles"].str.contains("baseline")].head(1)
        actual = group[group["treatment_observed"]].head(1)
        if baseline.empty or actual.empty:
            continue
        baseline_response = float(model.marginal_response(baseline)[0])
        actual_response = float(model.marginal_response(actual)[0])
        energy = float(actual["actual_energy_mwh"].iloc[0])
        records.append(
            {
                "decision_id": decision_id,
                "decision_at_utc": actual["decision_at_utc"].iloc[0],
                "horizon_hours": horizon,
                "actual_energy_mwh": energy,
                "baseline_marginal_response_kgco2e_per_mwh": baseline_response,
                "actual_marginal_response_kgco2e_per_mwh": actual_response,
                "avoided_emissions_point_kgco2e": energy
                * (baseline_response - actual_response),
            }
        )
    return pd.DataFrame(records)


def bootstrap_effects(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    horizon: int,
) -> pd.DataFrame:
    iterations = int(config["bootstrap_iterations"])
    if iterations <= 0:
        return pd.DataFrame()
    decision_times = train[["decision_id", "decision_at_utc"]].drop_duplicates(
        "decision_id"
    )
    block_days = int(config["bootstrap_block_days"])
    origin = pd.Timestamp(decision_times["decision_at_utc"].min()).floor("D")
    decision_times["block"] = (
        pd.to_datetime(decision_times["decision_at_utc"], utc=True) - origin
    ).dt.days // block_days
    block_ids = decision_times["block"].unique()
    if len(block_ids) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(int(config["random_seed"]) + int(horizon))
    draws = []
    for iteration in range(iterations):
        sampled_blocks = rng.choice(block_ids, size=len(block_ids), replace=True)
        pieces = []
        for sample_number, block in enumerate(sampled_blocks):
            ids = decision_times.loc[decision_times["block"].eq(block), "decision_id"]
            piece = train[train["decision_id"].isin(ids)].copy()
            piece["decision_id"] = (
                piece["decision_id"].astype(str) + f"-b{sample_number}"
            )
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        if sample["treatment_observed"].nunique() < 2:
            continue
        try:
            _, sample_weights, _ = fit_propensity(sample, test, config)
            model = ConstrainedMarginalResponseRegressor(
                list(config["adjustment_features"]),
                list(config["marginal_state_features"]),
                float(config["ridge_alpha"]),
            ).fit(
                sample,
                "outcome_interconnected_emissions_kgco2e",
                sample_weight=sample_weights,
            )
        except (RuntimeError, ValueError):
            continue
        effects = decision_effects(test, model, horizon)
        if effects.empty:
            continue
        effects["bootstrap_iteration"] = iteration
        draws.append(
            effects[
                [
                    "decision_id",
                    "bootstrap_iteration",
                    "actual_marginal_response_kgco2e_per_mwh",
                    "avoided_emissions_point_kgco2e",
                ]
            ]
        )
    return pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()


def attach_intervals(
    decisions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    output = decisions.copy()
    for level in config["confidence_levels"]:
        label = int(round(float(level) * 100))
        lower_q = (1 - float(level)) / 2
        upper_q = 1 - lower_q
        if bootstrap.empty:
            output[f"avoided_emissions_ci{label}_lower_kgco2e"] = np.nan
            output[f"avoided_emissions_ci{label}_upper_kgco2e"] = np.nan
            continue
        grouped = bootstrap.groupby("decision_id")["avoided_emissions_point_kgco2e"]
        lower = grouped.quantile(lower_q)
        upper = grouped.quantile(upper_q)
        output[f"avoided_emissions_ci{label}_lower_kgco2e"] = output["decision_id"].map(
            lower
        )
        output[f"avoided_emissions_ci{label}_upper_kgco2e"] = output["decision_id"].map(
            upper
        )
    confident_label = int(
        round(float(config["confidently_avoided_confidence_level"]) * 100)
    )
    lower = output[f"avoided_emissions_ci{confident_label}_lower_kgco2e"]
    output["confidently_avoided_kgco2e"] = lower.clip(lower=0).fillna(0.0)
    return output


def summarize_effects(
    decisions: pd.DataFrame,
    model: ConstrainedMarginalResponseRegressor,
    test: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    actual_states = test[test["treatment_observed"]]
    response = model.marginal_response(actual_states)
    result: dict[str, Any] = {
        "point_estimates": {
            "mean_marginal_response_kgco2e_per_mwh": float(np.mean(response)),
            "mean_avoided_emissions_kgco2e": safe_mean(
                decisions, "avoided_emissions_point_kgco2e"
            ),
            "total_confidently_avoided_kgco2e": float(
                decisions.get(
                    "confidently_avoided_kgco2e", pd.Series(dtype=float)
                ).sum()
            ),
        },
        "bootstrap_successful_iterations": int(
            bootstrap["bootstrap_iteration"].nunique() if not bootstrap.empty else 0
        ),
        "uncertainty_intervals": {},
    }
    if bootstrap.empty:
        return result
    iteration_means = bootstrap.groupby("bootstrap_iteration")[
        "avoided_emissions_point_kgco2e"
    ].mean()
    response_means = bootstrap.groupby("bootstrap_iteration")[
        "actual_marginal_response_kgco2e_per_mwh"
    ].mean()
    for level in config["confidence_levels"]:
        label = f"ci{int(round(float(level) * 100))}"
        lower_q = (1 - float(level)) / 2
        result["uncertainty_intervals"][label] = {
            "mean_marginal_response_lower_kgco2e_per_mwh": float(
                response_means.quantile(lower_q)
            ),
            "mean_marginal_response_upper_kgco2e_per_mwh": float(
                response_means.quantile(1 - lower_q)
            ),
            "mean_avoided_emissions_lower_kgco2e": float(
                iteration_means.quantile(lower_q)
            ),
            "mean_avoided_emissions_upper_kgco2e": float(
                iteration_means.quantile(1 - lower_q)
            ),
        }
    return result


def run_estimator(
    observations: pd.DataFrame,
    mart: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset = build_estimator_dataset(observations, mart, config)
    blockers = evaluate_estimator_readiness(observations, dataset, config)
    base_report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "estimator_version": config["estimator_version"],
        "uses_observed_workload_treatment": True,
        "uses_load_forecast_error_as_treatment": False,
        "adjustment_policy": "pre_treatment_only",
        "response_horizons_hours": config["response_horizons_hours"],
        "dataset_rows": int(len(dataset)),
        "completed_decisions": int(
            observations.get("status", pd.Series(dtype=str)).eq("completed").sum()
        ),
    }
    if blockers:
        return (
            {
                **base_report,
                "status": "blocked",
                "identified_estimator_ready": False,
                "evidence_tier": "not_identified_readiness_blocked",
                "blockers": blockers,
                "horizons": [],
            },
            dataset,
            pd.DataFrame(),
            {},
        )

    horizon_reports = []
    decision_outputs = []
    artifacts: dict[str, Any] = {}
    for horizon in config["response_horizons_hours"]:
        report, decisions, artifact = fit_horizon(dataset, int(horizon), config)
        horizon_reports.append(report)
        if not decisions.empty:
            decision_outputs.append(decisions)
        if artifact is not None:
            artifacts[str(horizon)] = artifact
    horizon_blockers = [
        f"horizon_{row['horizon_hours']}h:{blocker}"
        for row in horizon_reports
        if row["status"] != "ok"
        for blocker in row.get("blockers", [])
    ]
    ready = not horizon_blockers
    report = {
        **base_report,
        "status": "ok" if ready else "blocked",
        "identified_estimator_ready": ready,
        "evidence_tier": (
            "observational_identified_under_stated_assumptions"
            if ready
            else "not_identified_readiness_blocked"
        ),
        "blockers": horizon_blockers,
        "horizons": horizon_reports,
    }
    decisions = (
        pd.concat(decision_outputs, ignore_index=True)
        if decision_outputs
        else pd.DataFrame()
    )
    estimates = pd.DataFrame(
        [
            {
                "horizon_hours": row["horizon_hours"],
                **row.get("point_estimates", {}),
                **{
                    f"{interval}_{name}": value
                    for interval, values in row.get("uncertainty_intervals", {}).items()
                    for name, value in values.items()
                },
            }
            for row in horizon_reports
            if row["status"] == "ok"
        ]
    )
    return report, dataset, decisions, {"models": artifacts, "estimates": estimates}


def write_estimator_artifacts(
    report: dict[str, Any],
    dataset: pd.DataFrame,
    decisions: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    estimates_path: str | Path = DEFAULT_ESTIMATES_PATH,
    decisions_path: str | Path = DEFAULT_DECISIONS_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    paths = [
        Path(path)
        for path in (
            report_path,
            estimates_path,
            decisions_path,
            dataset_path,
            model_path,
        )
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dataset_path, index=False, compression="zstd")
    artifacts.get("estimates", pd.DataFrame()).to_csv(estimates_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    if report["identified_estimator_ready"]:
        joblib.dump(artifacts["models"], model_path)
    else:
        Path(model_path).unlink(missing_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


def dataset_columns(config: dict[str, Any]) -> list[str]:
    return [
        "decision_id",
        "decision_at_utc",
        "state_timestamp_utc",
        "state_roles",
        "horizon_hours",
        "actual_energy_mwh",
        "treatment_mwh",
        "treatment_observed",
        "eligible_for_fit",
        "point_in_time_eligible",
        "snapshot_at_utc",
        "information_cutoff_utc",
        "outcome_interconnected_emissions_kgco2e",
        "workload_type",
        "baseline_source",
        *config["adjustment_features"],
    ]


def parse_candidates(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def add_state_time(
    states: dict[pd.Timestamp, set[str]], timestamp: pd.Timestamp | None, role: str
) -> None:
    if timestamp is not None:
        states.setdefault(timestamp.floor("h"), set()).add(role)


def utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def intervals_overlap(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> bool:
    return left_start < right_end and right_start < left_end


def forward_outcome_sum(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float | None:
    timestamps = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz="UTC")
    column = "outcome_interconnected_direct_emissions_kgco2e_h0"
    if column not in hourly or not timestamps.isin(hourly.index).all():
        return None
    values = pd.to_numeric(hourly.loc[timestamps, column], errors="coerce")
    return float(values.sum()) if values.notna().all() else None


def safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if pd.notna(value) else None


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Fit the observed-treatment marginal emissions estimator."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mart-path", default=str(DEFAULT_MART_PATH))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")
    observations = load_treatment_observations(
        create_database_engine(args.database_url)
    )
    mart = pd.read_parquet(args.mart_path)
    config = load_estimator_config(args.config_path)
    report, dataset, decisions, artifacts = run_estimator(observations, mart, config)
    write_estimator_artifacts(report, dataset, decisions, artifacts)
    print(
        json.dumps(
            {
                "status": report["status"],
                "identified_estimator_ready": report["identified_estimator_ready"],
                "completed_decisions": report["completed_decisions"],
                "blockers": report["blockers"],
                "output": str(DEFAULT_REPORT_PATH),
            },
            indent=2,
        )
    )
    return 2 if args.require_ready and not report["identified_estimator_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
