"""Baseline forecasting models for France electricity spot prices."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "price_eur_mwh"
CONSUMPTION_TARGET_COLUMN = "consumption_mwh"
PRODUCTION_TARGET_COLUMN = "total_production_mwh"
TIMESTAMP_COLUMN = "timestamp_utc"
PRODUCTION_SIGNAL_TARGETS = [
    "production",
    "nuclear",
    "gas",
    "coal",
    "oil",
    "wind",
    "solar",
    "hydro",
    "bioenergy",
]

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_peak_hour",
    "is_morning_ramp",
    "is_evening_peak",
    "is_overnight",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

BASE_PRICE_FEATURES = [
    *CALENDAR_FEATURES,
    "price_lag_1h",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_72h",
    "price_lag_168h",
    "price_lag_336h",
    "price_rolling_mean_24h",
    "price_rolling_mean_168h",
    "price_rolling_std_24h",
    "price_rolling_min_24h",
    "price_rolling_max_24h",
    "price_rolling_range_24h",
    "price_lag_1h_to_24h",
    "price_lag_24h_to_168h",
    "price_lag_168h_to_336h",
    "price_vs_rolling_mean_24h",
    "consumption_lag_1h",
    "consumption_lag_24h",
    "consumption_lag_48h",
    "consumption_lag_72h",
    "consumption_lag_168h",
    "consumption_lag_336h",
    "consumption_rolling_mean_24h",
    "consumption_rolling_mean_168h",
    "consumption_rolling_std_24h",
    "consumption_rolling_min_24h",
    "consumption_rolling_max_24h",
    "consumption_rolling_range_24h",
    "consumption_lag_1h_to_24h",
    "consumption_lag_24h_to_168h",
    "total_production_lag_1h",
    "total_production_lag_24h",
    "total_production_lag_48h",
    "total_production_lag_72h",
    "total_production_lag_168h",
    "total_production_lag_336h",
    "total_production_rolling_mean_24h",
    "total_production_rolling_mean_168h",
    "total_production_rolling_std_24h",
    "total_production_rolling_min_24h",
    "total_production_rolling_max_24h",
    "total_production_rolling_range_24h",
    "total_production_lag_1h_to_24h",
    "total_production_lag_24h_to_168h",
    "wind_lag_24h",
    "wind_lag_168h",
    "wind_lag_24h_to_168h",
    "solar_lag_24h",
    "solar_lag_168h",
    "solar_lag_24h_to_168h",
    "residual_demand_lag_24h",
    "residual_demand_lag_168h",
    "residual_demand_lag_24h_to_168h",
    "variable_renewable_lag_24h",
    "variable_renewable_lag_168h",
    "variable_renewable_lag_24h_to_168h",
]

PRICE_FORECAST_SIGNAL_FEATURES = [
    "forecast_consumption_mwh",
    "forecast_total_production_mwh",
    "forecast_residual_demand_mwh",
    "forecast_supply_demand_gap_mwh",
]

STRICT_FORECAST_FEATURES = [*BASE_PRICE_FEATURES, *PRICE_FORECAST_SIGNAL_FEATURES]

SUPPLY_DEMAND_FEATURES = [
    *CALENDAR_FEATURES,
    "{prefix}_lag_1h",
    "{prefix}_lag_24h",
    "{prefix}_lag_48h",
    "{prefix}_lag_72h",
    "{prefix}_lag_168h",
    "{prefix}_lag_336h",
    "{prefix}_rolling_mean_24h",
    "{prefix}_rolling_mean_168h",
    "{prefix}_rolling_std_24h",
    "{prefix}_rolling_min_24h",
    "{prefix}_rolling_max_24h",
    "{prefix}_rolling_range_24h",
    "{prefix}_lag_1h_to_24h",
    "{prefix}_lag_24h_to_168h",
]

SUPPLY_DEMAND_TARGETS = {
    "consumption": (CONSUMPTION_TARGET_COLUMN, "consumption"),
    "production": (PRODUCTION_TARGET_COLUMN, "total_production"),
    "nuclear": ("nuclear_mwh", "nuclear"),
    "gas": ("gas_mwh", "gas"),
    "coal": ("coal_mwh", "coal"),
    "oil": ("oil_mwh", "oil"),
    "wind": ("wind_mwh", "wind"),
    "solar": ("solar_mwh", "solar"),
    "hydro": ("hydro_mwh", "hydro"),
    "bioenergy": ("bioenergy_mwh", "bioenergy"),
}


@dataclass(frozen=True)
class ForecastWindow:
    """One walk-forward validation/test window."""

    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DEFAULT_WALK_FORWARD_WINDOWS = [
    ForecastWindow("wf_2026_01", "2023-01-01", "2025-12-31 23:00:00+00:00", "2026-01-01", "2026-01-31 23:00:00+00:00"),
    ForecastWindow("wf_2026_02", "2023-01-01", "2026-01-31 23:00:00+00:00", "2026-02-01", "2026-02-28 23:00:00+00:00"),
    ForecastWindow("wf_2026_03", "2023-01-01", "2026-02-28 23:00:00+00:00", "2026-03-01", "2026-03-31 23:00:00+00:00"),
    ForecastWindow("test_2026_q2", "2023-01-01", "2026-03-31 23:00:00+00:00", "2026-04-01", "2026-06-30 23:00:00+00:00"),
]


def load_modeling_dataset(path: str | Path = "data/processed/modeling_price_features.csv") -> pd.DataFrame:
    """Load the feature dataset from CSV."""
    frame = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    return prepare_modeling_dataset(frame)


def prepare_modeling_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, type, and validate the modeling dataset."""
    required_targets = {TARGET_COLUMN, CONSUMPTION_TARGET_COLUMN, PRODUCTION_TARGET_COLUMN}
    if TIMESTAMP_COLUMN not in frame.columns or not required_targets.issubset(frame.columns):
        raise ValueError(f"Dataset must include {TIMESTAMP_COLUMN} and {sorted(required_targets)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        frame = add_missing_strict_features(frame)
    required_columns = {TIMESTAMP_COLUMN, TARGET_COLUMN, *BASE_PRICE_FEATURES}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required modeling columns: {sorted(missing_columns)}")

    prepared = frame.copy()
    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(prepared[TIMESTAMP_COLUMN], utc=True)
    prepared = prepared.sort_values(TIMESTAMP_COLUMN).drop_duplicates(TIMESTAMP_COLUMN)
    prepared = prepared.dropna(subset=[TARGET_COLUMN, *BASE_PRICE_FEATURES])

    if prepared.empty:
        raise ValueError("Modeling dataset is empty after required-column null filtering")

    return prepared


def add_missing_strict_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive strict forecasting features when loading an older feature export."""
    enriched = frame.copy()
    enriched[TIMESTAMP_COLUMN] = pd.to_datetime(enriched[TIMESTAMP_COLUMN], utc=True)
    enriched = enriched.sort_values(TIMESTAMP_COLUMN).drop_duplicates(TIMESTAMP_COLUMN)
    indexed = enriched.set_index(TIMESTAMP_COLUMN).asfreq("h")

    if "hour" not in indexed:
        indexed["hour"] = indexed.index.hour
    if "day_of_week" not in indexed:
        indexed["day_of_week"] = indexed.index.dayofweek
    if "month" not in indexed:
        indexed["month"] = indexed.index.month
    if "day_of_year" not in indexed:
        indexed["day_of_year"] = indexed.index.dayofyear
    if "is_weekend" not in indexed:
        indexed["is_weekend"] = indexed["day_of_week"].isin([5, 6])
    if "is_peak_hour" not in indexed:
        indexed["is_peak_hour"] = indexed["hour"].between(8, 20)
    if "is_morning_ramp" not in indexed:
        indexed["is_morning_ramp"] = indexed["hour"].between(6, 9)
    if "is_evening_peak" not in indexed:
        indexed["is_evening_peak"] = indexed["hour"].between(17, 20)
    if "is_overnight" not in indexed:
        indexed["is_overnight"] = indexed["hour"].isin([0, 1, 2, 3, 4, 5])
    if "hour_sin" not in indexed:
        indexed["hour_sin"] = np.sin(2 * np.pi * indexed["hour"] / 24)
    if "hour_cos" not in indexed:
        indexed["hour_cos"] = np.cos(2 * np.pi * indexed["hour"] / 24)
    if "day_of_year_sin" not in indexed:
        indexed["day_of_year_sin"] = np.sin(2 * np.pi * indexed["day_of_year"] / 365.25)
    if "day_of_year_cos" not in indexed:
        indexed["day_of_year_cos"] = np.cos(2 * np.pi * indexed["day_of_year"] / 365.25)

    indexed["price_lag_1h"] = indexed.get("price_lag_1h", indexed[TARGET_COLUMN].shift(1))
    indexed["price_lag_24h"] = indexed.get("price_lag_24h", indexed[TARGET_COLUMN].shift(24))
    indexed["price_lag_48h"] = indexed.get("price_lag_48h", indexed[TARGET_COLUMN].shift(48))
    indexed["price_lag_72h"] = indexed.get("price_lag_72h", indexed[TARGET_COLUMN].shift(72))
    indexed["price_lag_168h"] = indexed.get("price_lag_168h", indexed[TARGET_COLUMN].shift(168))
    indexed["price_lag_336h"] = indexed.get("price_lag_336h", indexed[TARGET_COLUMN].shift(336))
    indexed["price_rolling_mean_24h"] = indexed.get(
        "price_rolling_mean_24h",
        indexed[TARGET_COLUMN].shift(1).rolling(24).mean(),
    )
    indexed["price_rolling_mean_168h"] = indexed.get(
        "price_rolling_mean_168h",
        indexed[TARGET_COLUMN].shift(1).rolling(168).mean(),
    )
    indexed["price_rolling_std_24h"] = indexed.get(
        "price_rolling_std_24h",
        indexed[TARGET_COLUMN].shift(1).rolling(24).std(),
    )
    indexed["price_rolling_min_24h"] = indexed.get(
        "price_rolling_min_24h",
        indexed[TARGET_COLUMN].shift(1).rolling(24).min(),
    )
    indexed["price_rolling_max_24h"] = indexed.get(
        "price_rolling_max_24h",
        indexed[TARGET_COLUMN].shift(1).rolling(24).max(),
    )
    indexed["price_rolling_range_24h"] = indexed.get(
        "price_rolling_range_24h",
        indexed["price_rolling_max_24h"] - indexed["price_rolling_min_24h"],
    )
    indexed["price_lag_1h_to_24h"] = indexed.get(
        "price_lag_1h_to_24h",
        indexed["price_lag_1h"] - indexed["price_lag_24h"],
    )
    indexed["price_lag_24h_to_168h"] = indexed.get(
        "price_lag_24h_to_168h",
        indexed["price_lag_24h"] - indexed["price_lag_168h"],
    )
    indexed["price_lag_168h_to_336h"] = indexed.get(
        "price_lag_168h_to_336h",
        indexed["price_lag_168h"] - indexed["price_lag_336h"],
    )
    indexed["price_vs_rolling_mean_24h"] = indexed.get(
        "price_vs_rolling_mean_24h",
        indexed["price_lag_1h"] - indexed["price_rolling_mean_24h"],
    )
    for target_column, prefix in SUPPLY_DEMAND_TARGETS.values():
        if target_column in indexed:
            add_target_lag_features(indexed, target_column, prefix)

    for source_column, lag_column, periods in [
        ("consumption_mwh", "consumption_lag_24h", 24),
        ("consumption_mwh", "consumption_lag_168h", 168),
        ("wind_mwh", "wind_lag_24h", 24),
        ("wind_mwh", "wind_lag_168h", 168),
        ("solar_mwh", "solar_lag_24h", 24),
        ("solar_mwh", "solar_lag_168h", 168),
    ]:
        if lag_column not in indexed and source_column in indexed:
            indexed[lag_column] = indexed[source_column].shift(periods)

    indexed["consumption_lag_24h_to_168h"] = indexed.get(
        "consumption_lag_24h_to_168h",
        indexed["consumption_lag_24h"] - indexed["consumption_lag_168h"],
    )
    indexed["wind_lag_24h_to_168h"] = indexed.get(
        "wind_lag_24h_to_168h",
        indexed["wind_lag_24h"] - indexed["wind_lag_168h"],
    )
    indexed["solar_lag_24h_to_168h"] = indexed.get(
        "solar_lag_24h_to_168h",
        indexed["solar_lag_24h"] - indexed["solar_lag_168h"],
    )
    indexed["residual_demand_lag_24h"] = indexed.get(
        "residual_demand_lag_24h",
        indexed["consumption_lag_24h"]
        - indexed["wind_lag_24h"].fillna(0)
        - indexed["solar_lag_24h"].fillna(0),
    )
    indexed["residual_demand_lag_168h"] = indexed.get(
        "residual_demand_lag_168h",
        indexed["consumption_lag_168h"]
        - indexed["wind_lag_168h"].fillna(0)
        - indexed["solar_lag_168h"].fillna(0),
    )
    indexed["residual_demand_lag_24h_to_168h"] = indexed.get(
        "residual_demand_lag_24h_to_168h",
        indexed["residual_demand_lag_24h"] - indexed["residual_demand_lag_168h"],
    )
    indexed["variable_renewable_lag_24h"] = indexed.get(
        "variable_renewable_lag_24h",
        indexed["wind_lag_24h"].fillna(0) + indexed["solar_lag_24h"].fillna(0),
    )
    indexed["variable_renewable_lag_168h"] = indexed.get(
        "variable_renewable_lag_168h",
        indexed["wind_lag_168h"].fillna(0) + indexed["solar_lag_168h"].fillna(0),
    )
    indexed["variable_renewable_lag_24h_to_168h"] = indexed.get(
        "variable_renewable_lag_24h_to_168h",
        indexed["variable_renewable_lag_24h"] - indexed["variable_renewable_lag_168h"],
    )

    return indexed.reset_index()


def add_target_lag_features(frame: pd.DataFrame, source_column: str, prefix: str) -> None:
    """Derive lag and rolling features for one forecast target."""
    frame[f"{prefix}_lag_1h"] = frame.get(f"{prefix}_lag_1h", frame[source_column].shift(1))
    frame[f"{prefix}_lag_24h"] = frame.get(f"{prefix}_lag_24h", frame[source_column].shift(24))
    frame[f"{prefix}_lag_48h"] = frame.get(f"{prefix}_lag_48h", frame[source_column].shift(48))
    frame[f"{prefix}_lag_72h"] = frame.get(f"{prefix}_lag_72h", frame[source_column].shift(72))
    frame[f"{prefix}_lag_168h"] = frame.get(f"{prefix}_lag_168h", frame[source_column].shift(168))
    frame[f"{prefix}_lag_336h"] = frame.get(f"{prefix}_lag_336h", frame[source_column].shift(336))
    frame[f"{prefix}_rolling_mean_24h"] = frame.get(
        f"{prefix}_rolling_mean_24h",
        frame[source_column].shift(1).rolling(24).mean(),
    )
    frame[f"{prefix}_rolling_mean_168h"] = frame.get(
        f"{prefix}_rolling_mean_168h",
        frame[source_column].shift(1).rolling(168).mean(),
    )
    frame[f"{prefix}_rolling_std_24h"] = frame.get(
        f"{prefix}_rolling_std_24h",
        frame[source_column].shift(1).rolling(24).std(),
    )
    frame[f"{prefix}_rolling_min_24h"] = frame.get(
        f"{prefix}_rolling_min_24h",
        frame[source_column].shift(1).rolling(24).min(),
    )
    frame[f"{prefix}_rolling_max_24h"] = frame.get(
        f"{prefix}_rolling_max_24h",
        frame[source_column].shift(1).rolling(24).max(),
    )
    frame[f"{prefix}_rolling_range_24h"] = frame.get(
        f"{prefix}_rolling_range_24h",
        frame[f"{prefix}_rolling_max_24h"] - frame[f"{prefix}_rolling_min_24h"],
    )
    frame[f"{prefix}_lag_1h_to_24h"] = frame.get(
        f"{prefix}_lag_1h_to_24h",
        frame[f"{prefix}_lag_1h"] - frame[f"{prefix}_lag_24h"],
    )
    frame[f"{prefix}_lag_24h_to_168h"] = frame.get(
        f"{prefix}_lag_24h_to_168h",
        frame[f"{prefix}_lag_24h"] - frame[f"{prefix}_lag_168h"],
    )


def run_price_baselines(
    data_path: str | Path = "data/processed/modeling_price_features.csv",
    metrics_path: str | Path = "reports/metrics/price_baseline_metrics.json",
    predictions_path: str | Path = "reports/predictions/price_baseline_predictions.csv",
    diagnostics_path: str | Path = "reports/metrics/price_baseline_error_diagnostics.csv",
    top_errors_path: str | Path = "reports/metrics/price_baseline_top_errors.csv",
    feature_importance_path: str | Path = "reports/metrics/price_baseline_feature_importance.csv",
    ranking_path: str | Path = "reports/rankings/price_decision_rankings.csv",
    ranking_metrics_path: str | Path = "reports/metrics/price_ranking_metrics.json",
    supply_demand_metrics_path: str | Path = "reports/metrics/supply_demand_baseline_metrics.json",
    supply_demand_predictions_path: str | Path = "reports/predictions/supply_demand_baseline_predictions.csv",
    supply_demand_feature_importance_path: str | Path = (
        "reports/metrics/supply_demand_baseline_feature_importance.csv"
    ),
    artifacts_dir: str | Path = "models",
) -> dict[str, Any]:
    """Train and evaluate baseline models with walk-forward validation."""
    frame = load_modeling_dataset(data_path)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_feature_importance: list[pd.DataFrame] = []
    all_signal_metrics: list[dict[str, Any]] = []
    all_signal_predictions: list[pd.DataFrame] = []
    all_signal_feature_importance: list[pd.DataFrame] = []

    for window in DEFAULT_WALK_FORWARD_WINDOWS:
        train_frame, test_frame = split_window(frame, window)
        if train_frame.empty or test_frame.empty:
            raise ValueError(f"Empty train/test frame for window {window.name}")

        (
            train_frame,
            test_frame,
            signal_metrics,
            signal_predictions,
            signal_feature_importance,
        ) = add_supply_demand_forecasts(
            train_frame,
            test_frame,
            window,
            artifacts,
        )
        all_signal_metrics.extend(signal_metrics)
        all_signal_predictions.extend(signal_predictions)
        all_signal_feature_importance.extend(signal_feature_importance)

        for model_name, model in build_models().items():
            fitted_model = fit_model(model_name, model, train_frame)
            predictions = predict_model(model_name, fitted_model, test_frame)
            metrics = evaluate_predictions(
                actuals=test_frame[TARGET_COLUMN].to_numpy(),
                predictions=predictions,
            )
            all_metrics.append(
                {
                    "window": asdict(window),
                    "model": model_name,
                    "train_rows": int(len(train_frame)),
                    "test_rows": int(len(test_frame)),
                    **metrics,
                }
            )
            all_predictions.append(
                pd.DataFrame(
                    {
                        TIMESTAMP_COLUMN: test_frame[TIMESTAMP_COLUMN].values,
                        "window": window.name,
                        "model": model_name,
                        "actual_price_eur_mwh": test_frame[TARGET_COLUMN].values,
                        "predicted_price_eur_mwh": predictions,
                    }
                )
            )

            if window.name == "test_2026_q2" and model_name != "naive_lag_24h":
                joblib.dump(fitted_model, artifacts / f"{model_name}_price_baseline.joblib")
                importance = extract_feature_importance(model_name, fitted_model, window.name)
                if importance is not None:
                    all_feature_importance.append(importance)

    metrics_summary = summarize_metrics(all_metrics)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    diagnostics = build_error_diagnostics(predictions_frame)
    top_errors = build_top_error_periods(predictions_frame)
    rankings = build_price_decision_rankings(predictions_frame)
    ranking_metrics = summarize_ranking_metrics(rankings)
    feature_importance = combine_feature_importance(all_feature_importance)
    write_json(metrics_path, {"metrics": all_metrics, "summary": metrics_summary})
    write_predictions(predictions_path, predictions_frame)
    write_diagnostics(diagnostics_path, diagnostics)
    write_diagnostics(top_errors_path, top_errors)
    write_diagnostics(feature_importance_path, feature_importance)
    write_predictions(ranking_path, rankings)
    write_json(ranking_metrics_path, {"summary": ranking_metrics})
    signal_summary = summarize_signal_metrics(all_signal_metrics)
    write_json(
        supply_demand_metrics_path,
        {"metrics": all_signal_metrics, "summary": signal_summary},
    )
    write_predictions(
        supply_demand_predictions_path,
        pd.concat(all_signal_predictions, ignore_index=True),
    )
    signal_feature_importance_frame = combine_feature_importance(all_signal_feature_importance)
    write_diagnostics(supply_demand_feature_importance_path, signal_feature_importance_frame)
    return {"metrics": all_metrics, "summary": metrics_summary}


def run_price_ranking_from_predictions(
    predictions_path: str | Path = "reports/predictions/price_baseline_predictions.csv",
    ranking_path: str | Path = "reports/rankings/price_decision_rankings.csv",
    ranking_metrics_path: str | Path = "reports/metrics/price_ranking_metrics.json",
) -> dict[str, Any]:
    """Build decision rankings from saved price predictions."""
    predictions = pd.read_csv(predictions_path, parse_dates=[TIMESTAMP_COLUMN])
    rankings = build_price_decision_rankings(predictions)
    ranking_metrics = summarize_ranking_metrics(rankings)
    write_predictions(ranking_path, rankings)
    write_json(ranking_metrics_path, {"summary": ranking_metrics})
    return {"summary": ranking_metrics}


def split_window(frame: pd.DataFrame, window: ForecastWindow) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into one expanding-window train/test pair."""
    train_start = pd.Timestamp(window.train_start, tz="UTC")
    train_end = pd.Timestamp(window.train_end, tz="UTC")
    test_start = pd.Timestamp(window.test_start, tz="UTC")
    test_end = pd.Timestamp(window.test_end, tz="UTC")

    train_frame = frame[
        (frame[TIMESTAMP_COLUMN] >= train_start)
        & (frame[TIMESTAMP_COLUMN] <= train_end)
    ]
    test_frame = frame[
        (frame[TIMESTAMP_COLUMN] >= test_start)
        & (frame[TIMESTAMP_COLUMN] <= test_end)
    ]
    return train_frame, test_frame


def add_supply_demand_forecasts(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    window: ForecastWindow,
    artifacts: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    """Train supply/demand models and attach their forecasts for price modeling."""
    enriched_train = train_frame.copy()
    enriched_test = test_frame.copy()
    signal_metrics: list[dict[str, Any]] = []
    signal_predictions: list[pd.DataFrame] = []
    signal_feature_importance: list[pd.DataFrame] = []

    for signal_name in ["consumption", "production"]:
        target_result = evaluate_supply_demand_target(
            train_frame=train_frame,
            test_frame=test_frame,
            window=window,
            artifacts=artifacts,
            signal_name=signal_name,
        )
        signal_metrics.extend(target_result["metrics"])
        signal_predictions.extend(target_result["predictions"])
        signal_feature_importance.extend(target_result["feature_importance"])
        selected_train_predictions = target_result["selected_train_predictions"]
        selected_test_predictions = target_result["selected_test_predictions"]

        if selected_train_predictions is None or selected_test_predictions is None:
            raise ValueError(f"No selected forecast model was available for {signal_name}")

        if signal_name == "consumption":
            enriched_train["forecast_consumption_mwh"] = selected_train_predictions
            enriched_test["forecast_consumption_mwh"] = selected_test_predictions
        else:
            enriched_train["forecast_total_production_mwh"] = selected_train_predictions
            enriched_test["forecast_total_production_mwh"] = selected_test_predictions

    for enriched in [enriched_train, enriched_test]:
        enriched["forecast_residual_demand_mwh"] = (
            enriched["forecast_consumption_mwh"] - enriched["forecast_total_production_mwh"]
        )
        enriched["forecast_supply_demand_gap_mwh"] = (
            enriched["forecast_total_production_mwh"] - enriched["forecast_consumption_mwh"]
        )

    return enriched_train, enriched_test, signal_metrics, signal_predictions, signal_feature_importance


def evaluate_supply_demand_target(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    window: ForecastWindow,
    artifacts: Path,
    signal_name: str,
) -> dict[str, Any]:
    """Train and evaluate every model for one supply/demand target."""
    if signal_name not in SUPPLY_DEMAND_TARGETS:
        raise ValueError(f"Unknown supply/demand target: {signal_name}")

    target_column, prefix = SUPPLY_DEMAND_TARGETS[signal_name]
    feature_columns = supply_demand_feature_columns(prefix)
    models = build_models()
    selected_model_name = select_signal_model_name(models)
    selected_train_predictions: np.ndarray | None = None
    selected_test_predictions: np.ndarray | None = None
    target_metrics: list[dict[str, Any]] = []
    target_predictions: list[pd.DataFrame] = []
    target_feature_importance: list[pd.DataFrame] = []

    for model_name, model in models.items():
        fitted_model = fit_signal_model(model_name, model, train_frame, target_column, feature_columns)
        train_predictions = predict_signal_model(
            model_name,
            fitted_model,
            train_frame,
            feature_columns,
            prefix,
        )
        test_predictions = predict_signal_model(
            model_name,
            fitted_model,
            test_frame,
            feature_columns,
            prefix,
        )
        metrics = evaluate_predictions(
            actuals=test_frame[target_column].to_numpy(),
            predictions=test_predictions,
        )
        target_metrics.append(
            {
                "window": asdict(window),
                "target": signal_name,
                "model": model_name,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                **metrics,
            }
        )
        target_predictions.append(
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: test_frame[TIMESTAMP_COLUMN].values,
                    "window": window.name,
                    "target": signal_name,
                    "target_column": target_column,
                    "model": model_name,
                    "actual_mwh": np.round(test_frame[target_column].to_numpy(dtype=float), 2),
                    "predicted_mwh": np.round(test_predictions, 2),
                }
            )
        )

        if window.name == "test_2026_q2" and model_name != "naive_lag_24h":
            joblib.dump(fitted_model, artifacts / f"{model_name}_{signal_name}_baseline.joblib")
            importance = extract_feature_importance(
                model_name=model_name,
                model=fitted_model,
                window_name=window.name,
                feature_columns=feature_columns,
                target=signal_name,
            )
            if importance is not None:
                target_feature_importance.append(importance)

        if model_name == selected_model_name:
            selected_train_predictions = train_predictions
            selected_test_predictions = test_predictions

    return {
        "metrics": target_metrics,
        "predictions": target_predictions,
        "feature_importance": target_feature_importance,
        "selected_train_predictions": selected_train_predictions,
        "selected_test_predictions": selected_test_predictions,
    }


def run_supply_demand_baselines(
    target_names: list[str] | tuple[str, ...] = ("consumption", *PRODUCTION_SIGNAL_TARGETS),
    data_path: str | Path = "data/processed/modeling_price_features.csv",
    metrics_path: str | Path = "reports/metrics/supply_demand_baseline_metrics.json",
    predictions_path: str | Path = "reports/predictions/supply_demand_baseline_predictions.csv",
    feature_importance_path: str | Path = (
        "reports/metrics/supply_demand_baseline_feature_importance.csv"
    ),
    artifacts_dir: str | Path = "models",
) -> dict[str, Any]:
    """Train and evaluate consumption and/or production baselines."""
    frame = load_modeling_dataset(data_path)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_feature_importance: list[pd.DataFrame] = []
    for window in DEFAULT_WALK_FORWARD_WINDOWS:
        train_frame, test_frame = split_window(frame, window)
        if train_frame.empty or test_frame.empty:
            raise ValueError(f"Empty train/test frame for window {window.name}")

        for target_name in target_names:
            target_result = evaluate_supply_demand_target(
                train_frame=train_frame,
                test_frame=test_frame,
                window=window,
                artifacts=artifacts,
                signal_name=target_name,
            )
            all_metrics.extend(target_result["metrics"])
            all_predictions.extend(target_result["predictions"])
            all_feature_importance.extend(target_result["feature_importance"])

    metrics_summary = summarize_signal_metrics(all_metrics)
    write_json(metrics_path, {"metrics": all_metrics, "summary": metrics_summary})
    write_predictions(predictions_path, pd.concat(all_predictions, ignore_index=True))
    write_diagnostics(feature_importance_path, combine_feature_importance(all_feature_importance))
    return {"metrics": all_metrics, "summary": metrics_summary}


def supply_demand_feature_columns(prefix: str) -> list[str]:
    """Return concrete feature names for a supply/demand target prefix."""
    return [
        feature.format(prefix=prefix) if "{prefix}" in feature else feature
        for feature in SUPPLY_DEMAND_FEATURES
    ]


def select_signal_model_name(models: dict[str, Any]) -> str:
    """Select a fixed signal model without peeking at the test target."""
    for model_name in ["ridge", "lightgbm", "hist_gradient_boosting", "random_forest", "naive_lag_24h"]:
        if model_name in models:
            return model_name
    raise ValueError("No supply/demand model is available")


def fit_signal_model(
    model_name: str,
    model: Any,
    train_frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> Any:
    """Fit one supply/demand model."""
    if model_name == "naive_lag_24h":
        return None

    model.fit(train_frame[feature_columns], train_frame[target_column])
    return model


def predict_signal_model(
    model_name: str,
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    prefix: str,
) -> np.ndarray:
    """Predict one supply/demand target."""
    if model_name == "naive_lag_24h":
        return frame[f"{prefix}_lag_24h"].to_numpy(dtype=float)

    return model.predict(frame[feature_columns])


def build_models() -> dict[str, Any]:
    """Build baseline forecasting models."""
    models: dict[str, Any] = {
        "naive_lag_24h": None,
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=18,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    lightgbm_model = build_lightgbm_model()
    if lightgbm_model is not None:
        models["lightgbm"] = lightgbm_model

    xgboost_model = build_xgboost_model()
    if xgboost_model is not None:
        models["xgboost"] = xgboost_model

    return models


def build_lightgbm_model() -> Any | None:
    """Build LightGBM model when the optional dependency is installed."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=600,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_xgboost_model() -> Any | None:
    """Build XGBoost model when the optional dependency is installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=600,
                    learning_rate=0.03,
                    max_depth=6,
                    min_child_weight=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_model(model_name: str, model: Any, train_frame: pd.DataFrame) -> Any:
    """Fit one model, handling naive baselines separately."""
    if model_name == "naive_lag_24h":
        return None

    model.fit(train_frame[STRICT_FORECAST_FEATURES], train_frame[TARGET_COLUMN])
    return model


def predict_model(model_name: str, model: Any, test_frame: pd.DataFrame) -> np.ndarray:
    """Predict spot prices for one test frame."""
    if model_name == "naive_lag_24h":
        return test_frame["price_lag_24h"].to_numpy(dtype=float)

    return model.predict(test_frame[STRICT_FORECAST_FEATURES])


def evaluate_predictions(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Evaluate regression forecasts."""
    actuals = actuals.astype(float)
    predictions = predictions.astype(float)
    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions) ** 0.5
    smape = np.mean(
        2 * np.abs(predictions - actuals) / np.maximum(np.abs(actuals) + np.abs(predictions), 1e-9)
    )
    directional_accuracy = np.mean(
        np.sign(np.diff(actuals)) == np.sign(np.diff(predictions))
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
        "directional_accuracy": float(directional_accuracy),
    }


def summarize_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metrics by model across all walk-forward windows."""
    frame = pd.DataFrame(metrics)
    summary = (
        frame.groupby("model", as_index=False)[["mae", "rmse", "smape", "directional_accuracy"]]
        .mean()
        .sort_values("mae")
    )
    return summary.to_dict(orient="records")


def summarize_signal_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate supply/demand metrics by target and model."""
    frame = pd.DataFrame(metrics)
    summary = (
        frame.groupby(["target", "model"], as_index=False)[["mae", "rmse", "smape", "directional_accuracy"]]
        .mean()
        .sort_values("mae")
    )
    return summary.to_dict(orient="records")


def build_price_decision_rankings(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate hours by predicted price within each day/window/model."""
    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame["decision_date"] = frame[TIMESTAMP_COLUMN].dt.date.astype(str)
    group_columns = ["window", "model", "decision_date"]

    frame["predicted_price_rank"] = (
        frame.groupby(group_columns)["predicted_price_eur_mwh"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    frame["actual_price_rank"] = (
        frame.groupby(group_columns)["actual_price_eur_mwh"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    frame["candidate_count"] = frame.groupby(group_columns)[TIMESTAMP_COLUMN].transform("size")
    frame["actual_best_price_eur_mwh"] = (
        frame.groupby(group_columns)["actual_price_eur_mwh"].transform("min")
    )
    frame["regret_vs_best_eur_mwh"] = (
        frame["actual_price_eur_mwh"] - frame["actual_best_price_eur_mwh"]
    )
    frame["is_predicted_best"] = frame["predicted_price_rank"] == 1
    frame["is_actual_best"] = frame["actual_price_rank"] == 1
    frame["is_predicted_top_3"] = frame["predicted_price_rank"] <= 3
    frame["is_actual_top_3"] = frame["actual_price_rank"] <= 3
    return frame.sort_values(group_columns + ["predicted_price_rank"]).reset_index(drop=True)


def summarize_ranking_metrics(rankings: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize ranking quality by model across decision dates."""
    summaries: list[dict[str, Any]] = []
    for model_name, model_frame in rankings.groupby("model", observed=True):
        decision_groups = model_frame.groupby(["window", "decision_date"], observed=True)
        top_1_rows = model_frame[model_frame["predicted_price_rank"] == 1]
        top_3_rows = model_frame[model_frame["predicted_price_rank"] <= 3]
        actual_best_rows = model_frame[model_frame["actual_price_rank"] == 1]

        top_1_hit_rate = float(top_1_rows["is_actual_best"].mean())
        top_3_capture_rate = float(
            actual_best_rows.groupby(["window", "decision_date"], observed=True)
            .apply(lambda group: bool(group["is_predicted_top_3"].iloc[0]), include_groups=False)
            .mean()
        )
        mean_top_1_regret = float(top_1_rows["regret_vs_best_eur_mwh"].mean())
        median_top_1_regret = float(top_1_rows["regret_vs_best_eur_mwh"].median())
        mean_actual_rank_of_predicted_best = float(top_1_rows["actual_price_rank"].mean())
        mean_spearman = float(
            decision_groups.apply(
                lambda group: group["predicted_price_eur_mwh"].corr(
                    group["actual_price_eur_mwh"],
                    method="spearman",
                ),
                include_groups=False,
            ).mean()
        )
        top_3_average_actual_rank = float(top_3_rows["actual_price_rank"].mean())

        summaries.append(
            {
                "model": model_name,
                "decision_groups": int(decision_groups.ngroups),
                "top_1_hit_rate": top_1_hit_rate,
                "top_3_capture_rate": top_3_capture_rate,
                "mean_top_1_regret_eur_mwh": mean_top_1_regret,
                "median_top_1_regret_eur_mwh": median_top_1_regret,
                "mean_actual_rank_of_predicted_best": mean_actual_rank_of_predicted_best,
                "top_3_average_actual_rank": top_3_average_actual_rank,
                "mean_spearman_rank_corr": mean_spearman,
            }
        )

    return sorted(summaries, key=lambda row: row["mean_top_1_regret_eur_mwh"])


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_error_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build diagnostics by calendar bucket, price regime, and test window."""
    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = add_error_columns(frame)
    frame["hour"] = frame[TIMESTAMP_COLUMN].dt.hour
    frame["day_of_week"] = frame[TIMESTAMP_COLUMN].dt.dayofweek
    frame["month"] = frame[TIMESTAMP_COLUMN].dt.month
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6])
    frame["is_peak_hour"] = frame["hour"].between(8, 20)
    frame["is_morning_ramp"] = frame["hour"].between(6, 9)
    frame["is_evening_peak"] = frame["hour"].between(17, 20)
    frame["price_regime"] = pd.qcut(
        frame["actual_price_eur_mwh"],
        q=4,
        labels=["low", "medium_low", "medium_high", "high"],
        duplicates="drop",
    )

    diagnostics = []
    for group_name, group_columns in {
        "window": ["model", "window"],
        "hour": ["model", "hour"],
        "day_of_week": ["model", "day_of_week"],
        "month": ["model", "month"],
        "weekend": ["model", "is_weekend"],
        "peak_hour": ["model", "is_peak_hour"],
        "morning_ramp": ["model", "is_morning_ramp"],
        "evening_peak": ["model", "is_evening_peak"],
        "price_regime": ["model", "price_regime"],
    }.items():
        grouped = (
            frame.groupby(group_columns, observed=True)
            .agg(
                rows=("absolute_error", "size"),
                mean_error=("error", "mean"),
                mae=("absolute_error", "mean"),
                rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
                smape=("symmetric_absolute_percentage_error", "mean"),
                max_abs_error=("absolute_error", "max"),
                max_smape=("symmetric_absolute_percentage_error", "max"),
            )
            .reset_index()
        )
        grouped["diagnostic_group"] = group_name
        diagnostics.append(grouped)

    return pd.concat(diagnostics, ignore_index=True)


def add_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add signed, absolute, squared, and relative forecast errors."""
    enriched = frame.copy()
    enriched["error"] = (
        enriched["predicted_price_eur_mwh"] - enriched["actual_price_eur_mwh"]
    )
    enriched["absolute_error"] = enriched["error"].abs()
    enriched["squared_error"] = enriched["error"] ** 2
    denominator = (
        enriched["actual_price_eur_mwh"].abs()
        + enriched["predicted_price_eur_mwh"].abs()
    ).clip(lower=1e-9)
    enriched["symmetric_absolute_percentage_error"] = (
        2 * enriched["absolute_error"] / denominator
    )
    enriched["smape_pct"] = 100 * enriched["symmetric_absolute_percentage_error"]
    return enriched


def build_top_error_periods(predictions: pd.DataFrame, rows_per_model: int = 25) -> pd.DataFrame:
    """Return the largest absolute forecast misses for each model."""
    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = add_error_columns(frame)
    frame["hour"] = frame[TIMESTAMP_COLUMN].dt.hour
    frame["day_of_week"] = frame[TIMESTAMP_COLUMN].dt.dayofweek
    frame["is_peak_hour"] = frame["hour"].between(8, 20)
    frame["is_evening_peak"] = frame["hour"].between(17, 20)
    frame = frame.sort_values(["model", "absolute_error"], ascending=[True, False])
    return (
        frame.groupby("model", observed=True)
        .head(rows_per_model)
        .sort_values(["model", "absolute_error"], ascending=[True, False])
        .reset_index(drop=True)
    )


def combine_feature_importance(feature_importance: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine feature-importance frames with a stable empty schema."""
    if feature_importance:
        return pd.concat(feature_importance, ignore_index=True)

    return pd.DataFrame(columns=["window", "target", "model", "feature", "importance", "rank"])


def extract_feature_importance(
    model_name: str,
    model: Any,
    window_name: str,
    feature_columns: list[str] | None = None,
    target: str = "price",
) -> pd.DataFrame | None:
    """Extract feature importances or absolute coefficients from a fitted pipeline."""
    estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        return None

    importance = pd.DataFrame(
        {
            "window": window_name,
            "target": target,
            "model": model_name,
            "feature": feature_columns or STRICT_FORECAST_FEATURES,
            "importance": values,
        }
    )
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance


def write_predictions(path: str | Path, predictions: pd.DataFrame) -> None:
    """Write prediction artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)


def write_diagnostics(path: str | Path, diagnostics: pd.DataFrame) -> None:
    """Write error diagnostics artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output, index=False)
