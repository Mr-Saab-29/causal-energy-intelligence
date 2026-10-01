"""Build the leakage-aware hourly causal feature mart and readiness report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.carbon.intensity import load_emission_factor_config
from src.data.load import create_database_engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config/causal_feature_mart.json"
DEFAULT_EMISSION_FACTORS_PATH = ROOT / "config/emission_factors.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "reports/causal/causal_hourly_feature_mart.parquet"
DEFAULT_READINESS_PATH = ROOT / "reports/metrics/causal_feature_readiness.json"

ACTUAL_DISPATCH_COLUMNS = (
    "total_production_mwh",
    "nuclear_mwh",
    "thermal_mwh",
    "gas_mwh",
    "coal_mwh",
    "oil_mwh",
    "wind_mwh",
    "onshore_wind_mwh",
    "offshore_wind_mwh",
    "solar_mwh",
    "hydro_mwh",
    "pumped_storage_mwh",
    "bioenergy_mwh",
    "battery_storage_mwh",
    "physical_exchanges_mwh",
)
FORECAST_TYPES = ("load", "wind_onshore", "wind_offshore", "solar")
WEATHER_COLUMNS = (
    "temperature_c",
    "wind_speed_mps",
    "wind_speed_80m_mps",
    "shortwave_radiation_wm2",
    "cloud_cover_pct",
    "precipitation_mm",
)
EMISSIONS_SOURCE_COLUMNS = {
    "nuclear": "nuclear_mwh",
    "gas": "gas_mwh",
    "coal": "coal_mwh",
    "oil": "oil_mwh",
    "wind": "wind_mwh",
    "solar": "solar_mwh",
    "hydro": "hydro_mwh",
    "bioenergy": "bioenergy_mwh",
}


def build_and_write_feature_mart(
    engine: Engine,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    readiness_path: str | Path = DEFAULT_READINESS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load compact sources, build the mart, and write deterministic artifacts."""
    config = load_feature_contract(config_path)
    factors_by_methodology = load_emission_factor_config(DEFAULT_EMISSION_FACTORS_PATH)
    emission_factors = factors_by_methodology[config["outcome"]["methodology"]]
    sources = load_feature_sources(
        engine,
        pd.Timestamp(config["history_start_date"], tz="UTC"),
    )
    mart = build_feature_mart(
        sources["france"],
        sources["cross_border"],
        sources["weather"],
        sources["neighbor_emissions"],
        response_horizons=tuple(config["response_horizons_hours"]),
        contract_version=config["contract_version"],
        emission_factors=emission_factors,
        expected_neighbor_count=len(config["outcome"]["neighboring_areas"]),
        minimum_neighbor_factor_coverage=float(
            config["outcome"]["minimum_neighbor_factor_coverage"]
        ),
    )
    report = build_readiness_report(mart, config)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(destination, index=False, compression="zstd")
    report["artifacts"] = {
        "feature_mart": str(destination),
        "feature_mart_bytes": destination.stat().st_size,
        "readiness_report": str(readiness_path),
    }
    readiness = Path(readiness_path)
    readiness.parent.mkdir(parents=True, exist_ok=True)
    readiness.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return mart, report


def load_feature_contract(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "contract_version",
        "history_start_date",
        "minimum_core_coverage",
        "response_horizons_hours",
        "role_prefixes",
        "treatment_proxy",
        "outcome",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("feature mart contract missing: " + ", ".join(missing))
    if any(int(horizon) <= 0 for horizon in config["response_horizons_hours"]):
        raise ValueError("response horizons must be positive hours")
    return config


def load_feature_sources(
    engine: Engine,
    start: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Read bounded hourly sources without loading archived native rows."""
    params = {"start": start.to_pydatetime()}
    france_sql = text(
        """
        select
            causal.timestamp_utc,
            causal.vintage_quality,
            causal.load_forecast_mw,
            causal.wind_onshore_forecast_mw,
            causal.wind_offshore_forecast_mw,
            causal.solar_forecast_mw,
            causal.planned_unavailable_capacity_mw,
            causal.unplanned_unavailable_capacity_mw,
            causal.planned_outage_count,
            causal.unplanned_outage_count,
            causal.imbalance_price_up_eur_mwh,
            causal.imbalance_price_down_eur_mwh,
            causal.imbalance_volume_mwh,
            causal.activated_energy_price_eur_mwh,
            actual.consumption_mwh,
            actual.total_production_mwh,
            actual.nuclear_mwh,
            actual.thermal_mwh,
            actual.gas_mwh,
            actual.coal_mwh,
            actual.oil_mwh,
            actual.wind_mwh,
            actual.onshore_wind_mwh,
            actual.offshore_wind_mwh,
            actual.solar_mwh,
            actual.hydro_mwh,
            actual.pumped_storage_mwh,
            actual.bioenergy_mwh,
            actual.battery_storage_mwh,
            actual.physical_exchanges_mwh,
            actual.carbon_intensity_gco2_kwh,
            price.price_eur_mwh
        from causal_france_hourly_features as causal
        left join hourly_electricity_mix as actual
          on actual.region = 'FR'
         and actual.scope = 'national'
         and actual.timestamp_utc = causal.timestamp_utc
        left join (
            select
                date_trunc('hour', timestamp_utc) as timestamp_utc,
                avg(price_eur_mwh) as price_eur_mwh
            from electricity_prices
            where region = 'FR'
              and market = 'day_ahead'
            group by date_trunc('hour', timestamp_utc)
        ) as price
          on price.timestamp_utc = causal.timestamp_utc
        where causal.vintage_quality = 'historical_final'
          and causal.timestamp_utc >= :start
        order by causal.timestamp_utc
        """
    )
    cross_sql = text(
        """
        select
            timestamp_utc,
            sum(physical_import_mw) as physical_import_mw,
            sum(physical_export_mw) as physical_export_mw,
            sum(net_physical_import_mw) as net_physical_import_mw,
            sum(scheduled_import_mw) as scheduled_import_mw,
            sum(scheduled_export_mw) as scheduled_export_mw,
            sum(net_scheduled_import_mw) as net_scheduled_import_mw,
            sum(day_ahead_import_capacity_mw) as day_ahead_import_capacity_mw,
            sum(day_ahead_export_capacity_mw) as day_ahead_export_capacity_mw,
            count(*) filter (
                where physical_import_mw is not null or physical_export_mw is not null
            ) as physical_neighbor_count,
            count(*) filter (
                where scheduled_import_mw is not null or scheduled_export_mw is not null
            ) as scheduled_neighbor_count,
            count(*) as neighbor_count
        from causal_cross_border_hourly_features
        where vintage_quality = 'historical_final'
          and timestamp_utc >= :start
        group by timestamp_utc
        order by timestamp_utc
        """
    )
    weather_sql = text(
        """
        select
            date_trunc('hour', timestamp_utc) as timestamp_utc,
            avg(temperature_c) as temperature_c,
            avg(wind_speed_mps) as wind_speed_mps,
            avg(wind_speed_80m_mps) as wind_speed_80m_mps,
            avg(shortwave_radiation_wm2) as shortwave_radiation_wm2,
            avg(cloud_cover_pct) as cloud_cover_pct,
            avg(precipitation_mm) as precipitation_mm,
            count(distinct region) as weather_region_count
        from weather_observations
        where timestamp_utc >= :start
        group by date_trunc('hour', timestamp_utc)
        order by timestamp_utc
        """
    )
    neighbor_emissions_sql = text(
        """
        select
            timestamp_utc,
            bidding_zone,
            total_generation_mwh,
            named_factor_coverage_share,
            direct_emissions_kgco2e,
            direct_emissions_lower_kgco2e,
            direct_emissions_upper_kgco2e
        from causal_neighbor_hourly_emissions
        where vintage_quality = 'historical_final'
          and timestamp_utc >= :start
        order by timestamp_utc, bidding_zone
        """
    )
    with engine.connect() as connection:
        return {
            "france": pd.read_sql_query(france_sql, connection, params=params),
            "cross_border": pd.read_sql_query(cross_sql, connection, params=params),
            "weather": pd.read_sql_query(weather_sql, connection, params=params),
            "neighbor_emissions": pd.read_sql_query(
                neighbor_emissions_sql, connection, params=params
            ),
        }


def build_feature_mart(
    france: pd.DataFrame,
    cross_border: pd.DataFrame,
    weather: pd.DataFrame,
    neighbor_emissions: pd.DataFrame | None = None,
    *,
    response_horizons: tuple[int, ...] = (6, 12, 24),
    contract_version: str = "causal_hourly_feature_mart_v1",
    emission_factors: dict[str, float] | None = None,
    expected_neighbor_count: int = 6,
    minimum_neighbor_factor_coverage: float = 0.95,
) -> pd.DataFrame:
    """Create one row per decision hour with explicit causal roles."""
    if france.empty:
        raise ValueError("causal France compact features are empty")
    frame = france.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    frame = frame.merge(prepare_timestamp_frame(cross_border), on="timestamp_utc", how="left")
    frame = frame.merge(prepare_timestamp_frame(weather), on="timestamp_utc", how="left")
    frame = frame.merge(
        prepare_neighbor_emissions(neighbor_emissions),
        on="timestamp_utc",
        how="left",
    )
    frame = coerce_numeric_columns(frame)
    if emission_factors is None:
        emission_factors = load_emission_factor_config(DEFAULT_EMISSION_FACTORS_PATH)[
            "direct_operational_emissions"
        ]
    missing_factors = sorted(set(EMISSIONS_SOURCE_COLUMNS) - set(emission_factors))
    if missing_factors:
        raise ValueError("missing direct emission factors: " + ", ".join(missing_factors))

    mart = pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"],
            "metadata_contract_version": contract_version,
            "metadata_vintage_quality": frame["vintage_quality"],
            "metadata_point_in_time_eligible": frame["vintage_quality"].eq(
                "operational_snapshot"
            ),
        }
    )
    timestamp = frame["timestamp_utc"]
    mart["pre_hour"] = timestamp.dt.hour
    mart["pre_day_of_week"] = timestamp.dt.dayofweek
    mart["pre_month"] = timestamp.dt.month
    mart["pre_is_weekend"] = timestamp.dt.dayofweek.ge(5)
    mart["pre_hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    mart["pre_hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    mart["pre_day_of_year_sin"] = np.sin(
        2 * np.pi * timestamp.dt.dayofyear / 365.25
    )
    mart["pre_day_of_year_cos"] = np.cos(
        2 * np.pi * timestamp.dt.dayofyear / 365.25
    )

    for forecast_type in FORECAST_TYPES:
        source = f"{forecast_type}_forecast_mw"
        mart[f"pre_{source}"] = frame[source]
        actual = actual_column_for_forecast(forecast_type)
        error = frame[actual] - frame[source]
        mart[f"pre_{forecast_type}_forecast_error_lag_1h_mw"] = error.shift(1)
        mart[f"pre_{forecast_type}_forecast_error_lag_24h_mw"] = error.shift(24)
        mart[f"diagnostic_{forecast_type}_forecast_error_mw"] = error

    mart["pre_renewable_forecast_mw"] = frame[
        [
            "wind_onshore_forecast_mw",
            "wind_offshore_forecast_mw",
            "solar_forecast_mw",
        ]
    ].sum(axis=1, min_count=1)
    mart["pre_planned_unavailable_capacity_mw"] = frame[
        "planned_unavailable_capacity_mw"
    ]
    mart["pre_planned_outage_count"] = frame["planned_outage_count"]
    mart["pre_scheduled_import_mw"] = frame["scheduled_import_mw"]
    mart["pre_scheduled_export_mw"] = frame["scheduled_export_mw"]
    mart["pre_net_scheduled_import_mw"] = frame["net_scheduled_import_mw"]
    mart["pre_day_ahead_import_capacity_mw"] = frame[
        "day_ahead_import_capacity_mw"
    ]
    mart["pre_day_ahead_export_capacity_mw"] = frame[
        "day_ahead_export_capacity_mw"
    ]
    mart["pre_day_ahead_price_eur_mwh"] = frame["price_eur_mwh"]

    for column in WEATHER_COLUMNS:
        mart[f"pre_weather_{column}_lag_1h"] = frame[column].shift(1)
        mart[f"pre_weather_{column}_lag_24h"] = frame[column].shift(24)
    mart["pre_weather_region_count_lag_1h"] = frame["weather_region_count"].shift(1)

    mart["treatment_proxy_load_innovation_mw"] = (
        frame["consumption_mwh"] - frame["load_forecast_mw"]
    )
    source_emissions = []
    for source, column in EMISSIONS_SOURCE_COLUMNS.items():
        emissions = frame[column].clip(lower=0) * float(emission_factors[source])
        mart[f"outcome_france_{source}_direct_emissions_kgco2e_h0"] = emissions
        source_emissions.append(emissions)
    mart["outcome_france_direct_emissions_kgco2e_h0"] = pd.concat(
        source_emissions, axis=1
    ).sum(axis=1, min_count=len(source_emissions))
    mart["outcome_france_direct_carbon_intensity_gco2_kwh"] = (
        mart["outcome_france_direct_emissions_kgco2e_h0"]
        / frame["total_production_mwh"].replace(0, np.nan)
    )
    mart["outcome_neighbor_direct_emissions_kgco2e_h0"] = frame[
        "neighbor_direct_emissions_kgco2e"
    ]
    mart["outcome_neighbor_direct_emissions_lower_kgco2e_h0"] = frame[
        "neighbor_direct_emissions_lower_kgco2e"
    ]
    mart["outcome_neighbor_direct_emissions_upper_kgco2e_h0"] = frame[
        "neighbor_direct_emissions_upper_kgco2e"
    ]
    boundary_complete = frame["neighbor_zone_count"].eq(expected_neighbor_count)
    factor_coverage_meets_threshold = frame[
        "neighbor_min_named_factor_coverage"
    ].ge(
        minimum_neighbor_factor_coverage
    )
    mart["metadata_interconnected_boundary_complete"] = boundary_complete
    mart["metadata_neighbor_factor_coverage_meets_hourly_threshold"] = (
        factor_coverage_meets_threshold
    )
    mart["metadata_neighbor_zone_count"] = frame["neighbor_zone_count"]
    mart["metadata_neighbor_min_named_factor_coverage"] = frame[
        "neighbor_min_named_factor_coverage"
    ]
    neighbor_point = frame["neighbor_direct_emissions_kgco2e"].where(boundary_complete)
    neighbor_lower = frame["neighbor_direct_emissions_lower_kgco2e"].where(
        boundary_complete
    )
    neighbor_upper = frame["neighbor_direct_emissions_upper_kgco2e"].where(
        boundary_complete
    )
    france_emissions = mart["outcome_france_direct_emissions_kgco2e_h0"]
    mart["outcome_interconnected_direct_emissions_kgco2e_h0"] = (
        france_emissions + neighbor_point
    )
    mart["outcome_interconnected_direct_emissions_lower_kgco2e_h0"] = (
        france_emissions + neighbor_lower
    )
    mart["outcome_interconnected_direct_emissions_upper_kgco2e_h0"] = (
        france_emissions + neighbor_upper
    )
    for horizon in response_horizons:
        mart[f"outcome_france_direct_emissions_kgco2e_0_{horizon}h"] = forward_sum(
            mart["outcome_france_direct_emissions_kgco2e_h0"], horizon
        )
        for suffix in ("", "_lower", "_upper"):
            source = f"outcome_interconnected_direct_emissions{suffix}_kgco2e_h0"
            target = (
                f"outcome_interconnected_direct_emissions{suffix}_kgco2e_0_{horizon}h"
            )
            mart[target] = forward_sum(mart[source], horizon)

    mart = mart.copy()
    mart["mediator_physical_import_mw"] = frame["physical_import_mw"]
    mart["mediator_physical_export_mw"] = frame["physical_export_mw"]
    mart["mediator_net_physical_import_mw"] = frame["net_physical_import_mw"]
    mart["mediator_imbalance_price_up_eur_mwh"] = frame[
        "imbalance_price_up_eur_mwh"
    ]
    mart["mediator_imbalance_price_down_eur_mwh"] = frame[
        "imbalance_price_down_eur_mwh"
    ]
    mart["mediator_imbalance_volume_mwh"] = frame["imbalance_volume_mwh"]
    mart["mediator_activated_energy_price_eur_mwh"] = frame[
        "activated_energy_price_eur_mwh"
    ]
    for column in ACTUAL_DISPATCH_COLUMNS:
        mart[f"mediator_actual_{column}"] = frame[column]
    mart["diagnostic_actual_consumption_mwh"] = frame["consumption_mwh"]
    mart["diagnostic_odre_carbon_intensity_gco2_kwh"] = frame[
        "carbon_intensity_gco2_kwh"
    ]
    mart["diagnostic_unplanned_unavailable_capacity_mw"] = frame[
        "unplanned_unavailable_capacity_mw"
    ]
    mart["diagnostic_unplanned_outage_count"] = frame["unplanned_outage_count"]
    mart["diagnostic_physical_neighbor_count"] = frame["physical_neighbor_count"]
    mart["diagnostic_scheduled_neighbor_count"] = frame["scheduled_neighbor_count"]

    return add_missingness_indicators(mart)


def build_readiness_report(
    mart: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Separate artifact readiness from causal-identification readiness."""
    if mart.empty:
        raise ValueError("feature mart is empty")
    timestamp = pd.to_datetime(mart["timestamp_utc"], utc=True)
    expected_hours = int((timestamp.max() - timestamp.min()).total_seconds() // 3600) + 1
    duplicate_timestamps = int(timestamp.duplicated().sum())
    hourly_coverage = len(mart) / expected_hours if expected_hours else 0.0
    coverage = {
        column: round(float(mart[column].notna().mean()), 4)
        for column in mart.columns
        if column != "timestamp_utc"
    }
    core_columns = (
        "pre_load_forecast_mw",
        "pre_wind_onshore_forecast_mw",
        "pre_solar_forecast_mw",
        "pre_net_scheduled_import_mw",
        "treatment_proxy_load_innovation_mw",
        "outcome_france_direct_emissions_kgco2e_h0",
        "outcome_interconnected_direct_emissions_kgco2e_h0",
    )
    threshold = float(config["minimum_core_coverage"])
    core_failures = [
        f"low_core_coverage:{column}:{coverage.get(column, 0.0)}"
        for column in core_columns
        if coverage.get(column, 0.0) < threshold
    ]
    mart_issues = list(core_failures)
    if duplicate_timestamps:
        mart_issues.append(f"duplicate_timestamps:{duplicate_timestamps}")
    if hourly_coverage < 0.99:
        mart_issues.append(f"low_hourly_coverage:{hourly_coverage:.4f}")

    adjustment_columns = sorted(
        column for column in mart.columns if column.startswith("pre_")
    )
    forbidden_adjustment_columns = [
        column
        for column in adjustment_columns
        if column.startswith(("mediator_", "diagnostic_", "outcome_"))
    ]
    boundary_coverage = float(
        mart["metadata_interconnected_boundary_complete"].fillna(False).mean()
    )
    interconnected_boundary_ready = boundary_coverage >= threshold
    identification_blockers = [
        "historical_pre_treatment_inputs_are_not_point_in_time_operational_vintages",
        "historical_weather_forecasts_unavailable",
        "fuel_price_history_unavailable",
        "carbon_price_history_unavailable",
        "initial_storage_state_unavailable",
        "observed_workload_intervention_unavailable_treatment_is_proxy",
    ]
    if not interconnected_boundary_ready:
        identification_blockers.append("neighbor_emissions_outcome_incomplete")
    warnings = []
    low_neighbor_factor_hour_share = float(
        (~mart["metadata_neighbor_factor_coverage_meets_hourly_threshold"].fillna(False)).mean()
    )
    if low_neighbor_factor_hour_share:
        warnings.append(
            "neighbor_factor_fallback_high_hour_share:"
            f"{low_neighbor_factor_hour_share:.4f}"
        )
    for column in (
        "pre_wind_offshore_forecast_mw",
        "pre_day_ahead_import_capacity_mw",
        "pre_weather_temperature_c_lag_1h",
        "mediator_imbalance_volume_mwh",
    ):
        if coverage.get(column, 0.0) < threshold:
            warnings.append(f"limited_optional_coverage:{column}:{coverage.get(column, 0.0)}")

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contract_version": config["contract_version"],
        "status": "fail" if mart_issues else "warn",
        "feature_mart_ready": not mart_issues,
        "identified_estimator_ready": False,
        "interconnected_estimand_ready": False,
        "interconnected_boundary_ready": interconnected_boundary_ready,
        "window": {
            "start_utc": timestamp.min().isoformat(),
            "end_utc": timestamp.max().isoformat(),
            "rows": int(len(mart)),
            "expected_hours": expected_hours,
            "hourly_coverage": round(hourly_coverage, 4),
        },
        "quality": {
            "duplicate_timestamps": duplicate_timestamps,
            "core_issues": mart_issues,
            "warnings": warnings,
            "column_coverage": coverage,
        },
        "temporal_safety": {
            "adjustment_prefix": "pre_",
            "adjustment_columns": adjustment_columns,
            "forbidden_adjustment_columns": forbidden_adjustment_columns,
            "post_treatment_prefixes": ["mediator_", "diagnostic_", "outcome_"],
            "contemporaneous_forecast_errors_excluded_from_adjustment": True,
        },
        "readiness": {
            "descriptive_and_proxy_analysis": not mart_issues,
            "france_observational_estimator": False,
            "interconnected_causal_estimator": False,
            "blocking_gaps": identification_blockers,
        },
        "emissions_boundary": {
            "complete_hour_share": round(boundary_coverage, 4),
            "low_named_factor_coverage_hour_share": round(
                low_neighbor_factor_hour_share, 4
            ),
            "required_neighbor_count": len(config["outcome"]["neighboring_areas"]),
            "interconnected_boundary_ready": interconnected_boundary_ready,
        },
        "treatment_proxy": config["treatment_proxy"],
        "outcome": config["outcome"],
    }


def prepare_timestamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    return result.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")


def prepare_neighbor_emissions(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "timestamp_utc",
        "neighbor_zone_count",
        "neighbor_total_generation_mwh",
        "neighbor_min_named_factor_coverage",
        "neighbor_direct_emissions_kgco2e",
        "neighbor_direct_emissions_lower_kgco2e",
        "neighbor_direct_emissions_upper_kgco2e",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    return (
        result.groupby("timestamp_utc", as_index=False)
        .agg(
            neighbor_zone_count=("bidding_zone", "nunique"),
            neighbor_total_generation_mwh=("total_generation_mwh", "sum"),
            neighbor_min_named_factor_coverage=(
                "named_factor_coverage_share",
                "min",
            ),
            neighbor_direct_emissions_kgco2e=(
                "direct_emissions_kgco2e",
                "sum",
            ),
            neighbor_direct_emissions_lower_kgco2e=(
                "direct_emissions_lower_kgco2e",
                "sum",
            ),
            neighbor_direct_emissions_upper_kgco2e=(
                "direct_emissions_upper_kgco2e",
                "sum",
            ),
        )
        .loc[:, columns]
    )


def coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    excluded = {"timestamp_utc", "vintage_quality"}
    for column in result.columns:
        if column not in excluded:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def actual_column_for_forecast(forecast_type: str) -> str:
    return {
        "load": "consumption_mwh",
        "wind_onshore": "onshore_wind_mwh",
        "wind_offshore": "offshore_wind_mwh",
        "solar": "solar_mwh",
    }[forecast_type]


def forward_sum(series: pd.Series, periods: int) -> pd.Series:
    return series.iloc[::-1].rolling(periods, min_periods=periods).sum().iloc[::-1]


def add_missingness_indicators(mart: pd.DataFrame) -> pd.DataFrame:
    tracked = [
        "pre_wind_onshore_forecast_mw",
        "pre_wind_offshore_forecast_mw",
        "pre_solar_forecast_mw",
        "pre_day_ahead_import_capacity_mw",
        "pre_day_ahead_export_capacity_mw",
        "pre_weather_temperature_c_lag_1h",
        "mediator_imbalance_volume_mwh",
    ]
    indicators = {
        f"metadata_missing_{column}": mart[column].isna() for column in tracked
    }
    return pd.concat([mart, pd.DataFrame(indicators, index=mart.index)], axis=1)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build the hourly causal feature mart and readiness report."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--readiness-path", default=str(DEFAULT_READINESS_PATH))
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")

    mart, report = build_and_write_feature_mart(
        create_database_engine(args.database_url),
        config_path=args.config_path,
        output_path=args.output_path,
        readiness_path=args.readiness_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "feature_mart_ready": report["feature_mart_ready"],
                "identified_estimator_ready": report["identified_estimator_ready"],
                "rows": len(mart),
                "output": args.output_path,
                "readiness": args.readiness_path,
            },
            indent=2,
        )
    )
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
