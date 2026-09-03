"""Marginal-emissions proxy from hourly source-level generation changes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.carbon.intensity import DEFAULT_SOURCE_TARGETS, TIMESTAMP_COLUMN, load_emission_factor_config

DEFAULT_INPUT_PATH = "reports/carbon/hourly_carbon_intensity.csv"
DEFAULT_OUTPUT_PATH = "reports/carbon/marginal_emissions_proxy.csv"
DEFAULT_METRICS_PATH = "reports/metrics/marginal_emissions_proxy_metrics.json"
DEFAULT_EMISSION_FACTORS_PATH = "config/emission_factors.yaml"
MIN_RESPONSE_MWH = 1.0
MEDIUM_RESPONSE_MWH = 100.0
HIGH_RESPONSE_MWH = 500.0


def build_marginal_emissions_proxy(
    hourly_carbon: pd.DataFrame,
    emission_factors_by_methodology: dict[str, dict[str, float]],
    source_targets: tuple[str, ...] = DEFAULT_SOURCE_TARGETS,
    min_response_mwh: float = MIN_RESPONSE_MWH,
) -> pd.DataFrame:
    """Estimate hourly marginal intensity from positive source generation deltas."""
    required_columns = {TIMESTAMP_COLUMN, "methodology", "window", "model"}
    missing_columns = required_columns.difference(hourly_carbon.columns)
    if missing_columns:
        raise ValueError(f"Hourly carbon data is missing required columns: {sorted(missing_columns)}")

    outputs: list[pd.DataFrame] = []
    frame = hourly_carbon.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    for group_key, group in frame.groupby(["methodology", "window", "model"], observed=True):
        methodology = str(group_key[0])
        factors = emission_factors_by_methodology.get(methodology)
        if not factors:
            raise ValueError(f"No emission factors found for methodology {methodology!r}")
        for basis in ("actual", "predicted"):
            generation_columns = source_generation_columns(group, basis, source_targets)
            average_column = f"{basis}_carbon_intensity_g_co2e_per_kwh"
            outputs.append(
                build_group_proxy(
                    group=group,
                    generation_columns=generation_columns,
                    factors=factors,
                    source_targets=source_targets,
                    basis=basis,
                    average_column=average_column,
                    min_response_mwh=min_response_mwh,
                )
            )

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["methodology", "window", "model", "basis", TIMESTAMP_COLUMN]
    )


def build_group_proxy(
    group: pd.DataFrame,
    generation_columns: dict[str, str],
    factors: dict[str, float],
    source_targets: tuple[str, ...],
    basis: str,
    average_column: str,
    min_response_mwh: float,
) -> pd.DataFrame:
    """Build marginal proxy rows for one methodology/window/model/basis group."""
    ordered = group.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    generation = ordered[[generation_columns[source] for source in source_targets]].astype(float)
    generation.columns = list(source_targets)
    deltas = generation.diff()
    positive_response = deltas.clip(lower=0)
    total_response = positive_response.sum(axis=1)
    response_emissions = positive_response.mul(pd.Series(factors, dtype=float), axis="columns")
    marginal_intensity = response_emissions.sum(axis=1) / total_response.replace(0, np.nan)
    marginal_intensity = marginal_intensity.where(total_response >= min_response_mwh)

    output = ordered[[TIMESTAMP_COLUMN, "methodology", "window", "model"]].copy()
    output["basis"] = basis
    output["total_generation_mwh"] = generation.sum(axis=1)
    output["total_generation_delta_mwh"] = output["total_generation_mwh"].diff()
    output["total_positive_response_mwh"] = total_response
    output["average_carbon_intensity_g_co2e_per_kwh"] = ordered.get(average_column)
    output["marginal_carbon_intensity_g_co2e_per_kwh"] = marginal_intensity
    output["marginal_minus_average_g_co2e_per_kwh"] = (
        output["marginal_carbon_intensity_g_co2e_per_kwh"]
        - output["average_carbon_intensity_g_co2e_per_kwh"]
    )

    response_shares = positive_response.div(total_response.replace(0, np.nan), axis="rows")
    dominant_source = response_shares.fillna(-1).idxmax(axis=1).where(
        total_response >= min_response_mwh
    )
    output["marginal_source"] = dominant_source
    output["marginal_source_emission_factor_kg_co2e_per_mwh"] = dominant_source.map(factors)
    output["marginal_proxy_confidence"] = [
        confidence_label(response_mwh, source_share)
        for response_mwh, source_share in zip(total_response, response_shares.max(axis=1), strict=True)
    ]

    for source in source_targets:
        output[f"{source}_response_mwh"] = positive_response[source]
        output[f"{source}_response_share"] = response_shares[source]

    return output


def source_generation_columns(
    frame: pd.DataFrame,
    basis: str,
    source_targets: tuple[str, ...],
) -> dict[str, str]:
    """Return source generation columns for a data basis, validating availability."""
    columns = {
        source: f"{basis}_{source}_generation_mwh"
        for source in source_targets
    }
    missing = [column for column in columns.values() if column not in frame]
    if missing:
        raise ValueError(f"Hourly carbon data is missing generation columns: {missing}")
    return columns


def confidence_label(response_mwh: float, dominant_share: float) -> str:
    """Label proxy reliability from response size and concentration."""
    if pd.isna(response_mwh) or response_mwh < MIN_RESPONSE_MWH:
        return "unavailable"
    if response_mwh >= HIGH_RESPONSE_MWH and dominant_share >= 0.5:
        return "high"
    if response_mwh >= MEDIUM_RESPONSE_MWH:
        return "medium"
    return "low"


def summarize_marginal_proxy(proxy: pd.DataFrame) -> dict[str, Any]:
    """Summarize marginal proxy coverage and intensity diagnostics."""
    if proxy.empty:
        return {"summary": [], "source_mix": []}

    usable = proxy["marginal_carbon_intensity_g_co2e_per_kwh"].notna()
    summary = (
        proxy.assign(usable=usable)
        .groupby(["methodology", "window", "model", "basis"], as_index=False)
        .agg(
            rows=(TIMESTAMP_COLUMN, "count"),
            usable_rows=("usable", "sum"),
            usable_share=("usable", "mean"),
            mean_marginal_intensity_g_co2e_per_kwh=(
                "marginal_carbon_intensity_g_co2e_per_kwh",
                "mean",
            ),
            median_marginal_intensity_g_co2e_per_kwh=(
                "marginal_carbon_intensity_g_co2e_per_kwh",
                "median",
            ),
            mean_average_intensity_g_co2e_per_kwh=(
                "average_carbon_intensity_g_co2e_per_kwh",
                "mean",
            ),
            mean_marginal_minus_average_g_co2e_per_kwh=(
                "marginal_minus_average_g_co2e_per_kwh",
                "mean",
            ),
        )
    )
    source_mix = (
        proxy.loc[usable & proxy["marginal_source"].notna()]
        .groupby(["methodology", "window", "model", "basis", "marginal_source"], as_index=False)
        .size()
        .rename(columns={"size": "hours"})
    )
    if not source_mix.empty:
        source_mix["hour_share"] = source_mix["hours"] / source_mix.groupby(
            ["methodology", "window", "model", "basis"]
        )["hours"].transform("sum")

    return {
        "summary": sanitize_json_value(summary.to_dict(orient="records")),
        "source_mix": sanitize_json_value(source_mix.to_dict(orient="records")),
    }


def run_marginal_emissions_proxy(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    emission_factors_path: str | Path = DEFAULT_EMISSION_FACTORS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Build and persist marginal-emissions proxy outputs."""
    hourly_carbon = pd.read_csv(input_path, parse_dates=[TIMESTAMP_COLUMN])
    factors = load_emission_factor_config(emission_factors_path)
    proxy = build_marginal_emissions_proxy(hourly_carbon, factors)
    write_csv(output_path, proxy)
    metrics = summarize_marginal_proxy(proxy)
    metrics.update(
        {
            "generated_from": {
                "hourly_carbon_intensity": str(input_path),
                "emission_factors": str(emission_factors_path),
            },
            "output": str(output_path),
            "rows": int(len(proxy)),
        }
    )
    write_json(metrics_path, metrics)
    return metrics


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV file, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write strict JSON, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sanitize_json_value(value: Any) -> Any:
    """Return a value that can be emitted as strict JSON."""
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return sanitize_json_value(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main(argv: list[str] | None = None) -> None:
    """Run marginal-emissions proxy generation from the command line."""
    parser = argparse.ArgumentParser(description="Build marginal-emissions proxy artifacts.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--emission-factors-path", default=DEFAULT_EMISSION_FACTORS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    args = parser.parse_args(argv)

    result = run_marginal_emissions_proxy(
        input_path=args.input_path,
        emission_factors_path=args.emission_factors_path,
        output_path=args.output_path,
        metrics_path=args.metrics_path,
    )
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "output": result["output"],
                "summary_rows": len(result["summary"]),
                "source_mix_rows": len(result["source_mix"]),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
