"""Baseline forecasting models for France electricity spot prices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "price_eur_mwh"
TIMESTAMP_COLUMN = "timestamp_utc"

STRICT_FORECAST_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
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
    "consumption_lag_24h",
    "consumption_lag_168h",
    "wind_lag_24h",
    "wind_lag_168h",
    "solar_lag_24h",
    "solar_lag_168h",
]


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
    if TIMESTAMP_COLUMN not in frame.columns or TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must include {TIMESTAMP_COLUMN} and {TARGET_COLUMN}")

    frame = add_missing_strict_features(frame)
    required_columns = {TIMESTAMP_COLUMN, TARGET_COLUMN, *STRICT_FORECAST_FEATURES}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required modeling columns: {sorted(missing_columns)}")

    prepared = frame.copy()
    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(prepared[TIMESTAMP_COLUMN], utc=True)
    prepared = prepared.sort_values(TIMESTAMP_COLUMN).drop_duplicates(TIMESTAMP_COLUMN)
    prepared = prepared.dropna(subset=[TARGET_COLUMN, *STRICT_FORECAST_FEATURES])

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

    return indexed.reset_index()


def run_price_baselines(
    data_path: str | Path = "data/processed/modeling_price_features.csv",
    metrics_path: str | Path = "reports/metrics/price_baseline_metrics.json",
    predictions_path: str | Path = "reports/predictions/price_baseline_predictions.csv",
    diagnostics_path: str | Path = "reports/metrics/price_baseline_error_diagnostics.csv",
    artifacts_dir: str | Path = "models",
) -> dict[str, Any]:
    """Train and evaluate baseline models with walk-forward validation."""
    frame = load_modeling_dataset(data_path)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for window in DEFAULT_WALK_FORWARD_WINDOWS:
        train_frame, test_frame = split_window(frame, window)
        if train_frame.empty or test_frame.empty:
            raise ValueError(f"Empty train/test frame for window {window.name}")

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

    metrics_summary = summarize_metrics(all_metrics)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    diagnostics = build_error_diagnostics(predictions_frame)
    write_json(metrics_path, {"metrics": all_metrics, "summary": metrics_summary})
    write_predictions(predictions_path, predictions_frame)
    write_diagnostics(diagnostics_path, diagnostics)
    return {"metrics": all_metrics, "summary": metrics_summary}


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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_error_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build diagnostics by hour, weekday, and price regime."""
    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame["absolute_error"] = (
        frame["predicted_price_eur_mwh"] - frame["actual_price_eur_mwh"]
    ).abs()
    frame["squared_error"] = (
        frame["predicted_price_eur_mwh"] - frame["actual_price_eur_mwh"]
    ) ** 2
    frame["hour"] = frame[TIMESTAMP_COLUMN].dt.hour
    frame["day_of_week"] = frame[TIMESTAMP_COLUMN].dt.dayofweek
    frame["price_regime"] = pd.qcut(
        frame["actual_price_eur_mwh"],
        q=4,
        labels=["low", "medium_low", "medium_high", "high"],
        duplicates="drop",
    )

    diagnostics = []
    for group_name, group_columns in {
        "hour": ["model", "hour"],
        "day_of_week": ["model", "day_of_week"],
        "price_regime": ["model", "price_regime"],
    }.items():
        grouped = (
            frame.groupby(group_columns, observed=True)
            .agg(
                rows=("absolute_error", "size"),
                mae=("absolute_error", "mean"),
                rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            )
            .reset_index()
        )
        grouped["diagnostic_group"] = group_name
        diagnostics.append(grouped)

    return pd.concat(diagnostics, ignore_index=True)


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
