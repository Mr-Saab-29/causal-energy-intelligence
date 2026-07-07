"""Build strict forecasting features for France electricity spot prices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

MODELING_FEATURE_COLUMNS = [
    "timestamp_utc",
    "price_eur_mwh",
    "consumption_mwh",
    "total_production_mwh",
    "nuclear_mwh",
    "thermal_mwh",
    "gas_mwh",
    "coal_mwh",
    "oil_mwh",
    "wind_mwh",
    "solar_mwh",
    "hydro_mwh",
    "bioenergy_mwh",
    "physical_exchanges_mwh",
    "renewable_mwh",
    "fossil_mwh",
    "renewable_share",
    "nuclear_share",
    "fossil_share",
    "residual_demand_mwh",
    "supply_demand_gap_mwh",
    "avg_temperature_c",
    "min_temperature_c",
    "max_temperature_c",
    "avg_apparent_temperature_c",
    "avg_wind_speed_mps",
    "avg_wind_speed_80m_mps",
    "avg_shortwave_radiation_wm2",
    "avg_cloud_cover_pct",
    "avg_precipitation_mm",
    "total_precipitation_mm",
    "avg_surface_pressure_hpa",
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
    "price_lag_168h",
    "price_rolling_mean_24h",
    "price_rolling_mean_168h",
    "consumption_lag_24h",
    "wind_lag_24h",
    "solar_lag_24h",
]


def fetch_base_price_frame(engine: Engine) -> pd.DataFrame:
    """Fetch national price/load/generation data plus compact weather aggregates."""
    query = text(
        """
        with national_mix as (
            select *
            from hourly_electricity_mix
            where scope = 'national'
              and region = 'FR'
        )
        select
            price.timestamp_utc,
            price.price_eur_mwh,
            mix.consumption_mwh,
            mix.total_production_mwh,
            mix.nuclear_mwh,
            mix.thermal_mwh,
            mix.gas_mwh,
            mix.coal_mwh,
            mix.oil_mwh,
            mix.wind_mwh,
            mix.solar_mwh,
            mix.hydro_mwh,
            mix.bioenergy_mwh,
            mix.physical_exchanges_mwh,
            weather.avg_temperature_c,
            weather.min_temperature_c,
            weather.max_temperature_c,
            weather.avg_apparent_temperature_c,
            weather.avg_wind_speed_mps,
            weather.avg_wind_speed_80m_mps,
            weather.avg_shortwave_radiation_wm2,
            weather.avg_cloud_cover_pct,
            weather.avg_precipitation_mm,
            weather.total_precipitation_mm,
            weather.avg_surface_pressure_hpa
        from electricity_prices as price
        inner join national_mix as mix
            on mix.timestamp_utc = price.timestamp_utc
        left join weather_france_hourly_agg as weather
            on weather.timestamp_utc = price.timestamp_utc
        where price.region = 'FR'
          and price.market = 'day_ahead'
        order by price.timestamp_utc
        """
    )
    return pd.read_sql_query(query, engine)


def build_price_modeling_features(base_frame: pd.DataFrame) -> pd.DataFrame:
    """Build strict forecasting and causal-ready price features."""
    frame = base_frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    frame = frame.set_index("timestamp_utc").asfreq("h")

    numeric_columns = [column for column in frame.columns if column != "timestamp_utc"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    _add_power_mix_features(frame)
    _add_time_features(frame)
    _add_lag_features(frame)

    frame = frame.reset_index()
    frame = frame.dropna(
        subset=[
            "price_eur_mwh",
            "consumption_mwh",
            "total_production_mwh",
            "price_lag_168h",
            "price_rolling_mean_168h",
            "consumption_lag_24h",
            "wind_lag_24h",
            "solar_lag_24h",
        ]
    )
    return frame[MODELING_FEATURE_COLUMNS]


def export_price_modeling_features(
    engine: Engine,
    output_path: str | Path = "data/processed/modeling_price_features.csv",
) -> pd.DataFrame:
    """Build features from Supabase and export them to CSV."""
    features = build_price_modeling_features(fetch_base_price_frame(engine))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return features


def upsert_price_modeling_features(engine: Engine, features: pd.DataFrame, batch_size: int = 1_000) -> int:
    """Persist modeling features into Supabase."""
    if features.empty:
        return 0

    rows = [_row_to_database_dict(row) for row in features.to_dict(orient="records")]
    columns = MODELING_FEATURE_COLUMNS
    insert_columns = ", ".join(columns)
    value_columns = ", ".join(f":{column}" for column in columns)
    update_assignments = ", ".join(
        f"{column} = excluded.{column}" for column in columns if column != "timestamp_utc"
    )
    statement = text(
        f"""
        insert into modeling_price_features ({insert_columns})
        values ({value_columns})
        on conflict (timestamp_utc) do update set {update_assignments}
        """
    )

    with engine.begin() as connection:
        for start_index in range(0, len(rows), batch_size):
            connection.execute(statement, rows[start_index : start_index + batch_size])

    return len(rows)


def build_and_store_price_modeling_features(
    engine: Engine,
    output_path: str | Path = "data/processed/modeling_price_features.csv",
) -> pd.DataFrame:
    """Build, export, and persist price modeling features."""
    features = export_price_modeling_features(engine, output_path=output_path)
    upsert_price_modeling_features(engine, features)
    return features


def _add_power_mix_features(frame: pd.DataFrame) -> None:
    frame["renewable_mwh"] = (
        frame["wind_mwh"].fillna(0)
        + frame["solar_mwh"].fillna(0)
        + frame["hydro_mwh"].fillna(0)
        + frame["bioenergy_mwh"].fillna(0)
    )
    frame["fossil_mwh"] = (
        frame["thermal_mwh"].fillna(0)
        + frame["gas_mwh"].fillna(0)
        + frame["coal_mwh"].fillna(0)
        + frame["oil_mwh"].fillna(0)
    )
    frame["renewable_share"] = _safe_divide(frame["renewable_mwh"], frame["total_production_mwh"])
    frame["nuclear_share"] = _safe_divide(frame["nuclear_mwh"], frame["total_production_mwh"])
    frame["fossil_share"] = _safe_divide(frame["fossil_mwh"], frame["total_production_mwh"])
    frame["residual_demand_mwh"] = (
        frame["consumption_mwh"] - frame["wind_mwh"].fillna(0) - frame["solar_mwh"].fillna(0)
    )
    frame["supply_demand_gap_mwh"] = frame["total_production_mwh"] - frame["consumption_mwh"]


def _add_time_features(frame: pd.DataFrame) -> None:
    index = frame.index
    frame["hour"] = index.hour
    frame["day_of_week"] = index.dayofweek
    frame["month"] = index.month
    frame["day_of_year"] = index.dayofyear
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6])
    frame["is_peak_hour"] = frame["hour"].between(8, 20)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)


def _add_lag_features(frame: pd.DataFrame) -> None:
    frame["price_lag_1h"] = frame["price_eur_mwh"].shift(1)
    frame["price_lag_24h"] = frame["price_eur_mwh"].shift(24)
    frame["price_lag_168h"] = frame["price_eur_mwh"].shift(168)
    frame["price_rolling_mean_24h"] = frame["price_eur_mwh"].shift(1).rolling(24).mean()
    frame["price_rolling_mean_168h"] = frame["price_eur_mwh"].shift(1).rolling(168).mean()
    frame["consumption_lag_24h"] = frame["consumption_mwh"].shift(24)
    frame["wind_lag_24h"] = frame["wind_mwh"].shift(24)
    frame["solar_lag_24h"] = frame["solar_mwh"].shift(24)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _row_to_database_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None if pd.isna(value) else value
        for key, value in row.items()
    }
