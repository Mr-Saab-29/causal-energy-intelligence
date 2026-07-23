"""Carbon-intensity accounting from actual and forecast generation by source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

TIMESTAMP_COLUMN = "timestamp_utc"
DEFAULT_SOURCE_TARGETS = ("nuclear", "gas", "coal", "oil", "wind", "solar", "hydro", "bioenergy")


def load_emission_factor_config(path: str | Path) -> dict[str, dict[str, float]]:
    """Load carbon-accounting methodologies and source-level emission factors."""
    with Path(path).open() as file:
        config = yaml.safe_load(file)

    methodologies = config.get("methodologies", {}) if isinstance(config, dict) else {}
    if not methodologies:
        raise ValueError("Emission factor config must define at least one methodology")

    parsed: dict[str, dict[str, float]] = {}
    for methodology, payload in methodologies.items():
        factors = payload.get("emission_factors_kg_co2e_per_mwh", {})
        if not factors:
            raise ValueError(f"Methodology {methodology!r} has no emission factors")
        parsed[methodology] = {source: float(value) for source, value in factors.items()}
    return parsed


def build_carbon_outputs_from_predictions(
    predictions: pd.DataFrame,
    emission_factors_by_methodology: dict[str, dict[str, float]],
    source_targets: tuple[str, ...] = DEFAULT_SOURCE_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build hourly emissions, hourly source contributions, and evaluation metrics."""
    required_columns = {
        TIMESTAMP_COLUMN,
        "window",
        "target",
        "model",
        "actual_mwh",
        "predicted_mwh",
    }
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(f"Prediction data is missing required columns: {sorted(missing_columns)}")

    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = frame[frame["target"].isin(source_targets)]
    available_targets = set(frame["target"].unique())
    missing_targets = set(source_targets).difference(available_targets)
    if missing_targets:
        raise ValueError(
            "Prediction data must include source-level generation targets: "
            f"{sorted(missing_targets)}"
        )

    hourly_outputs: list[pd.DataFrame] = []
    contribution_outputs: list[pd.DataFrame] = []
    metrics_outputs: list[dict[str, Any]] = []

    group_columns = ["window", "model"]
    for methodology, factors in emission_factors_by_methodology.items():
        missing_factors = set(source_targets).difference(factors)
        if missing_factors:
            raise ValueError(
                f"Methodology {methodology!r} is missing factors for {sorted(missing_factors)}"
            )

        for (window, model), model_frame in frame.groupby(group_columns, observed=True):
            actual_generation = pivot_generation(model_frame, "actual_mwh", source_targets)
            predicted_generation = pivot_generation(model_frame, "predicted_mwh", source_targets)
            hourly = build_hourly_carbon_frame(
                actual_generation=actual_generation,
                predicted_generation=predicted_generation,
                factors=factors,
                methodology=methodology,
                window=str(window),
                model=str(model),
            )
            hourly_outputs.append(hourly)
            contribution_outputs.append(build_contribution_frame(hourly, source_targets))
            metrics_outputs.append(evaluate_carbon_outputs(hourly, methodology, str(window), str(model)))

    hourly_frame = pd.concat(hourly_outputs, ignore_index=True)
    contribution_frame = pd.concat(contribution_outputs, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics_outputs).sort_values(
        ["methodology", "window", "model"]
    )
    return hourly_frame, contribution_frame, metrics_frame.reset_index(drop=True)


def pivot_generation(frame: pd.DataFrame, value_column: str, source_targets: tuple[str, ...]) -> pd.DataFrame:
    """Pivot long-form source forecasts into one hourly generation table."""
    pivoted = frame.pivot_table(
        index=TIMESTAMP_COLUMN,
        columns="target",
        values=value_column,
        aggfunc="first",
    ).sort_index()
    missing = set(source_targets).difference(pivoted.columns)
    if missing:
        raise ValueError(f"Missing generation values for sources: {sorted(missing)}")
    return pivoted.loc[:, list(source_targets)].astype(float)


def build_hourly_carbon_frame(
    actual_generation: pd.DataFrame,
    predicted_generation: pd.DataFrame,
    factors: dict[str, float],
    methodology: str,
    window: str,
    model: str,
) -> pd.DataFrame:
    """Calculate hourly actual and predicted emissions and carbon intensity."""
    factor_series = pd.Series(factors, dtype=float)
    actual_generation = actual_generation.clip(lower=0)
    predicted_generation = predicted_generation.clip(lower=0)
    actual_source_emissions = actual_generation.mul(factor_series, axis="columns")
    predicted_source_emissions = predicted_generation.mul(factor_series, axis="columns")

    output = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: actual_generation.index,
            "methodology": methodology,
            "window": window,
            "model": model,
            "actual_total_generation_mwh": actual_generation.sum(axis=1).to_numpy(),
            "predicted_total_generation_mwh": predicted_generation.sum(axis=1).to_numpy(),
            "actual_total_emissions_kg_co2e": actual_source_emissions.sum(axis=1).to_numpy(),
            "predicted_total_emissions_kg_co2e": predicted_source_emissions.sum(axis=1).to_numpy(),
        }
    )
    output["actual_carbon_intensity_g_co2e_per_kwh"] = divide_or_nan(
        output["actual_total_emissions_kg_co2e"],
        output["actual_total_generation_mwh"],
    )
    output["predicted_carbon_intensity_g_co2e_per_kwh"] = divide_or_nan(
        output["predicted_total_emissions_kg_co2e"],
        output["predicted_total_generation_mwh"],
    )

    for source in actual_generation.columns:
        output[f"actual_{source}_generation_mwh"] = actual_generation[source].to_numpy()
        output[f"predicted_{source}_generation_mwh"] = predicted_generation[source].to_numpy()
        output[f"actual_{source}_emissions_kg_co2e"] = actual_source_emissions[source].to_numpy()
        output[f"predicted_{source}_emissions_kg_co2e"] = predicted_source_emissions[source].to_numpy()

    return output


def divide_or_nan(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while preserving NaN when hourly generation is zero."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def build_contribution_frame(hourly: pd.DataFrame, source_targets: tuple[str, ...]) -> pd.DataFrame:
    """Create long-form technology-level hourly emission contribution output."""
    records: list[pd.DataFrame] = []
    base_columns = [TIMESTAMP_COLUMN, "methodology", "window", "model"]
    for source in source_targets:
        contribution = hourly[base_columns].copy()
        contribution["source"] = source
        contribution["actual_generation_mwh"] = hourly[f"actual_{source}_generation_mwh"]
        contribution["predicted_generation_mwh"] = hourly[f"predicted_{source}_generation_mwh"]
        contribution["actual_emissions_kg_co2e"] = hourly[f"actual_{source}_emissions_kg_co2e"]
        contribution["predicted_emissions_kg_co2e"] = hourly[f"predicted_{source}_emissions_kg_co2e"]
        contribution["actual_emissions_share"] = divide_or_nan(
            contribution["actual_emissions_kg_co2e"],
            hourly["actual_total_emissions_kg_co2e"],
        )
        contribution["predicted_emissions_share"] = divide_or_nan(
            contribution["predicted_emissions_kg_co2e"],
            hourly["predicted_total_emissions_kg_co2e"],
        )
        records.append(contribution)
    return pd.concat(records, ignore_index=True)


def evaluate_carbon_outputs(
    hourly: pd.DataFrame,
    methodology: str,
    window: str,
    model: str,
) -> dict[str, Any]:
    """Evaluate predicted carbon outputs against actual carbon outputs."""
    emissions_errors = (
        hourly["predicted_total_emissions_kg_co2e"]
        - hourly["actual_total_emissions_kg_co2e"]
    )
    intensity_errors = (
        hourly["predicted_carbon_intensity_g_co2e_per_kwh"]
        - hourly["actual_carbon_intensity_g_co2e_per_kwh"]
    )
    return {
        "methodology": methodology,
        "window": window,
        "model": model,
        "rows": int(len(hourly)),
        "emissions_mae_kg_co2e": mae(emissions_errors),
        "emissions_rmse_kg_co2e": rmse(emissions_errors),
        "emissions_bias_kg_co2e": float(emissions_errors.mean()),
        "emissions_smape": smape(
            hourly["actual_total_emissions_kg_co2e"],
            hourly["predicted_total_emissions_kg_co2e"],
        ),
        "carbon_intensity_mae_g_co2e_per_kwh": mae(intensity_errors),
        "carbon_intensity_rmse_g_co2e_per_kwh": rmse(intensity_errors),
        "carbon_intensity_bias_g_co2e_per_kwh": float(intensity_errors.mean()),
        "carbon_intensity_smape": smape(
            hourly["actual_carbon_intensity_g_co2e_per_kwh"],
            hourly["predicted_carbon_intensity_g_co2e_per_kwh"],
        ),
    }


def mae(errors: pd.Series) -> float:
    """Mean absolute error for a prepared error series."""
    return float(np.nanmean(np.abs(errors)))


def rmse(errors: pd.Series) -> float:
    """Root mean squared error for a prepared error series."""
    return float(np.sqrt(np.nanmean(np.square(errors))))


def smape(actuals: pd.Series, predictions: pd.Series) -> float:
    """Symmetric mean absolute percentage error."""
    actual_values = actuals.to_numpy()
    prediction_values = predictions.to_numpy()
    denominator = np.maximum(np.abs(actual_values) + np.abs(prediction_values), 1e-9)
    return float(np.nanmean(2 * np.abs(prediction_values - actual_values) / denominator))


def run_carbon_accounting(
    predictions_path: str | Path = "reports/predictions/production_baseline_predictions.csv",
    emission_factors_path: str | Path = "config/emission_factors.yaml",
    hourly_output_path: str | Path = "reports/carbon/hourly_carbon_intensity.csv",
    contributions_output_path: str | Path = "reports/carbon/technology_emission_contributions.csv",
    metrics_output_path: str | Path = "reports/metrics/carbon_forecast_metrics.json",
) -> dict[str, Any]:
    """Run carbon accounting from saved source-level production predictions."""
    predictions = pd.read_csv(predictions_path, parse_dates=[TIMESTAMP_COLUMN])
    factors = load_emission_factor_config(emission_factors_path)
    hourly, contributions, metrics = build_carbon_outputs_from_predictions(predictions, factors)

    write_csv(hourly_output_path, hourly)
    write_csv(contributions_output_path, contributions)
    summary = summarize_carbon_metrics(metrics)
    write_json(
        metrics_output_path,
        {
            "metrics": metrics.to_dict(orient="records"),
            "summary": summary,
        },
    )
    return {
        "hourly_rows": int(len(hourly)),
        "contribution_rows": int(len(contributions)),
        "metrics": metrics.to_dict(orient="records"),
        "summary": summary,
    }


def summarize_carbon_metrics(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate carbon forecast metrics by methodology and model."""
    summary_columns = [
        "emissions_mae_kg_co2e",
        "emissions_rmse_kg_co2e",
        "emissions_bias_kg_co2e",
        "emissions_smape",
        "carbon_intensity_mae_g_co2e_per_kwh",
        "carbon_intensity_rmse_g_co2e_per_kwh",
        "carbon_intensity_bias_g_co2e_per_kwh",
        "carbon_intensity_smape",
    ]
    return (
        metrics.groupby(["methodology", "model"], as_index=False)[summary_columns]
        .mean()
        .sort_values(["methodology", "emissions_mae_kg_co2e"])
        .to_dict(orient="records")
    )


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV file, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
