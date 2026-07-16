from __future__ import annotations

import pandas as pd

from src.carbon.intensity import build_carbon_outputs_from_predictions


def test_build_carbon_outputs_from_predictions_calculates_hourly_totals_and_metrics() -> None:
    sources = ("gas", "wind")
    predictions = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "window": "test",
                "target": "gas",
                "model": "ridge",
                "actual_mwh": 10.0,
                "predicted_mwh": 12.0,
            },
            {
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "window": "test",
                "target": "wind",
                "model": "ridge",
                "actual_mwh": 30.0,
                "predicted_mwh": 28.0,
            },
            {
                "timestamp_utc": "2026-01-01T01:00:00Z",
                "window": "test",
                "target": "gas",
                "model": "ridge",
                "actual_mwh": 20.0,
                "predicted_mwh": 15.0,
            },
            {
                "timestamp_utc": "2026-01-01T01:00:00Z",
                "window": "test",
                "target": "wind",
                "model": "ridge",
                "actual_mwh": 20.0,
                "predicted_mwh": 25.0,
            },
        ]
    )
    factors = {"direct": {"gas": 400.0, "wind": 0.0}}

    hourly, contributions, metrics = build_carbon_outputs_from_predictions(
        predictions,
        factors,
        source_targets=sources,
    )

    first_hour = hourly.sort_values("timestamp_utc").iloc[0]
    assert first_hour["actual_total_generation_mwh"] == 40.0
    assert first_hour["predicted_total_generation_mwh"] == 40.0
    assert first_hour["actual_total_emissions_kg_co2e"] == 4000.0
    assert first_hour["predicted_total_emissions_kg_co2e"] == 4800.0
    assert first_hour["actual_carbon_intensity_g_co2e_per_kwh"] == 100.0
    assert first_hour["predicted_carbon_intensity_g_co2e_per_kwh"] == 120.0

    gas_contribution = contributions[
        (contributions["timestamp_utc"] == first_hour["timestamp_utc"])
        & (contributions["source"] == "gas")
    ].iloc[0]
    assert gas_contribution["actual_emissions_share"] == 1.0
    assert len(metrics) == 1
    assert metrics.iloc[0]["emissions_mae_kg_co2e"] == 1400.0
