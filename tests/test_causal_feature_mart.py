from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.causal.feature_mart import (
    build_feature_mart,
    build_readiness_report,
    load_feature_contract,
)


def make_sources(
    periods: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2026-08-01", periods=periods, freq="h", tz="UTC")
    france = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "vintage_quality": "historical_final",
            "load_forecast_mw": 98.0,
            "wind_onshore_forecast_mw": 18.0,
            "wind_offshore_forecast_mw": 4.0,
            "solar_forecast_mw": 8.0,
            "planned_unavailable_capacity_mw": 2.0,
            "unplanned_unavailable_capacity_mw": 1.0,
            "planned_outage_count": 1,
            "unplanned_outage_count": 1,
            "imbalance_price_up_eur_mwh": 70.0,
            "imbalance_price_down_eur_mwh": 30.0,
            "imbalance_volume_mwh": 3.0,
            "activated_energy_price_eur_mwh": 60.0,
            "consumption_mwh": 100.0,
            "total_production_mwh": 101.0,
            "nuclear_mwh": 50.0,
            "thermal_mwh": 12.0,
            "gas_mwh": 10.0,
            "coal_mwh": 1.0,
            "oil_mwh": 1.0,
            "wind_mwh": 24.0,
            "onshore_wind_mwh": 20.0,
            "offshore_wind_mwh": 4.0,
            "solar_mwh": 9.0,
            "hydro_mwh": 6.0,
            "pumped_storage_mwh": 0.0,
            "bioenergy_mwh": 0.0,
            "battery_storage_mwh": 0.0,
            "physical_exchanges_mwh": -1.0,
            "carbon_intensity_gco2_kwh": 50.0,
            "price_eur_mwh": 45.0,
        }
    )
    cross_border = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "physical_import_mw": 11.0,
            "physical_export_mw": 5.0,
            "net_physical_import_mw": 6.0,
            "scheduled_import_mw": 10.0,
            "scheduled_export_mw": 4.0,
            "net_scheduled_import_mw": 6.0,
            "day_ahead_import_capacity_mw": 20.0,
            "day_ahead_export_capacity_mw": 18.0,
            "physical_neighbor_count": 6,
            "scheduled_neighbor_count": 6,
            "neighbor_count": 6,
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "temperature_c": 20.0,
            "wind_speed_mps": 5.0,
            "wind_speed_80m_mps": 7.0,
            "shortwave_radiation_wm2": 300.0,
            "cloud_cover_pct": 30.0,
            "precipitation_mm": 0.0,
            "weather_region_count": 12,
        }
    )
    neighbor_emissions = pd.DataFrame(
        [
            {
                "timestamp_utc": timestamp,
                "bidding_zone": area,
                "total_generation_mwh": 100.0,
                "named_factor_coverage_share": 1.0,
                "direct_emissions_kgco2e": 20_000.0,
                "direct_emissions_lower_kgco2e": 20_000.0,
                "direct_emissions_upper_kgco2e": 20_000.0,
            }
            for timestamp in timestamps
            for area in ("BE", "DE_LU", "CH", "IT_NORD", "ES", "GB")
        ]
    )
    return france, cross_border, weather, neighbor_emissions


def test_feature_mart_separates_pre_treatment_and_post_treatment_roles() -> None:
    france, cross_border, weather, neighbor_emissions = make_sources()

    mart = build_feature_mart(france, cross_border, weather, neighbor_emissions)

    assert mart.loc[0, "treatment_proxy_load_innovation_mw"] == pytest.approx(2.0)
    assert np.isnan(mart.loc[0, "pre_load_forecast_error_lag_1h_mw"])
    assert mart.loc[1, "pre_load_forecast_error_lag_1h_mw"] == pytest.approx(2.0)
    assert mart.loc[0, "diagnostic_load_forecast_error_mw"] == pytest.approx(2.0)
    assert mart.loc[0, "pre_net_scheduled_import_mw"] == pytest.approx(6.0)
    assert mart.loc[0, "mediator_net_physical_import_mw"] == pytest.approx(6.0)
    assert "mediator_actual_consumption_mwh" not in mart
    assert "diagnostic_actual_consumption_mwh" in mart


def test_feature_mart_builds_exact_forward_response_windows() -> None:
    france, cross_border, weather, neighbor_emissions = make_sources()

    mart = build_feature_mart(france, cross_border, weather, neighbor_emissions)

    hourly_emissions = 10.0 * 370.0 + 1.0 * 820.0 + 1.0 * 650.0
    assert mart.loc[0, "outcome_france_direct_emissions_kgco2e_h0"] == hourly_emissions
    assert mart.loc[0, "outcome_france_direct_carbon_intensity_gco2_kwh"] == pytest.approx(
        hourly_emissions / 101.0
    )
    assert mart.loc[0, "diagnostic_odre_carbon_intensity_gco2_kwh"] == 50.0
    assert mart.loc[0, "outcome_neighbor_direct_emissions_kgco2e_h0"] == 120_000.0
    assert mart.loc[0, "outcome_interconnected_direct_emissions_kgco2e_h0"] == pytest.approx(
        hourly_emissions + 120_000.0
    )
    assert bool(mart.loc[0, "metadata_interconnected_boundary_complete"]) is True
    assert mart.loc[0, "outcome_france_direct_emissions_kgco2e_0_6h"] == pytest.approx(
        6 * hourly_emissions
    )
    assert mart["outcome_france_direct_emissions_kgco2e_0_6h"].isna().sum() == 5
    assert mart["outcome_france_direct_emissions_kgco2e_0_24h"].isna().sum() == 23


def test_readiness_distinguishes_mart_quality_from_identification() -> None:
    france, cross_border, weather, neighbor_emissions = make_sources()
    mart = build_feature_mart(france, cross_border, weather, neighbor_emissions)
    config = load_feature_contract()

    report = build_readiness_report(mart, config)

    assert report["status"] == "warn"
    assert report["feature_mart_ready"] is True
    assert report["identified_estimator_ready"] is False
    assert report["interconnected_boundary_ready"] is True
    assert report["readiness"]["descriptive_and_proxy_analysis"] is True
    assert "neighbor_emissions_outcome_incomplete" not in report["readiness"]["blocking_gaps"]
    assert report["temporal_safety"]["forbidden_adjustment_columns"] == []
    assert all(
        column.startswith("pre_")
        for column in report["temporal_safety"]["adjustment_columns"]
    )


def test_contract_rejects_missing_required_keys(tmp_path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"contract_version": "test"}), encoding="utf-8")

    with pytest.raises(ValueError, match="feature mart contract missing"):
        load_feature_contract(path)


def test_readiness_fails_when_one_neighbor_is_missing() -> None:
    france, cross_border, weather, neighbor_emissions = make_sources()
    neighbor_emissions = neighbor_emissions[
        neighbor_emissions["bidding_zone"] != "GB"
    ]
    mart = build_feature_mart(france, cross_border, weather, neighbor_emissions)

    report = build_readiness_report(mart, load_feature_contract())

    assert report["feature_mart_ready"] is False
    assert report["interconnected_boundary_ready"] is False
    assert "neighbor_emissions_outcome_incomplete" in report["readiness"][
        "blocking_gaps"
    ]


def test_low_hourly_factor_coverage_warns_without_masking_emissions() -> None:
    france, cross_border, weather, neighbor_emissions = make_sources()
    first_hour = neighbor_emissions["timestamp_utc"].min()
    neighbor_emissions.loc[
        (neighbor_emissions["timestamp_utc"] == first_hour)
        & (neighbor_emissions["bidding_zone"] == "IT_NORD"),
        "named_factor_coverage_share",
    ] = 0.5

    mart = build_feature_mart(france, cross_border, weather, neighbor_emissions)
    report = build_readiness_report(mart, load_feature_contract())

    assert bool(mart.loc[0, "metadata_interconnected_boundary_complete"]) is True
    assert bool(
        mart.loc[0, "metadata_neighbor_factor_coverage_meets_hourly_threshold"]
    ) is False
    assert pd.notna(
        mart.loc[0, "outcome_interconnected_direct_emissions_kgco2e_h0"]
    )
    assert report["feature_mart_ready"] is True
    assert report["interconnected_boundary_ready"] is True
    assert "neighbor_factor_fallback_high_hour_share:0.0333" in report["quality"][
        "warnings"
    ]
