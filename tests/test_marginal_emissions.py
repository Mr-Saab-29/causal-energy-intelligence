from __future__ import annotations

import json

import pandas as pd

from src.carbon.marginal import (
    build_marginal_emissions_proxy,
    run_marginal_emissions_proxy,
    summarize_marginal_proxy,
)


FACTORS = {
    "direct_operational_emissions": {
        "nuclear": 0.0,
        "gas": 370.0,
        "coal": 820.0,
        "oil": 650.0,
        "wind": 0.0,
        "solar": 0.0,
        "hydro": 0.0,
        "bioenergy": 230.0,
    }
}


def test_build_marginal_proxy_uses_positive_generation_response() -> None:
    proxy = build_marginal_emissions_proxy(sample_hourly_carbon(), FACTORS)
    actual = proxy[proxy["basis"] == "actual"].reset_index(drop=True)

    assert pd.isna(actual.loc[0, "marginal_carbon_intensity_g_co2e_per_kwh"])
    assert actual.loc[1, "marginal_source"] == "gas"
    assert actual.loc[1, "marginal_carbon_intensity_g_co2e_per_kwh"] == 370.0
    assert actual.loc[2, "marginal_source"] == "coal"
    assert actual.loc[2, "marginal_carbon_intensity_g_co2e_per_kwh"] == 525.0
    assert actual.loc[2, "coal_response_share"] == 0.5
    assert actual.loc[2, "bioenergy_response_share"] == 0.5


def test_summarize_marginal_proxy_reports_usable_coverage() -> None:
    proxy = build_marginal_emissions_proxy(sample_hourly_carbon(), FACTORS)
    summary = summarize_marginal_proxy(proxy)

    first = summary["summary"][0]
    assert first["rows"] == 3
    assert first["usable_rows"] == 2
    assert first["usable_share"] == 2 / 3
    assert first["mean_marginal_intensity_g_co2e_per_kwh"] == 447.5
    assert summary["source_mix"]


def test_run_marginal_proxy_writes_strict_json_outputs(tmp_path) -> None:
    input_path = tmp_path / "hourly_carbon.csv"
    output_path = tmp_path / "marginal.csv"
    metrics_path = tmp_path / "metrics.json"
    factors_path = tmp_path / "factors.yaml"

    sample_hourly_carbon().to_csv(input_path, index=False)
    factors_path.write_text(
        """
methodologies:
  direct_operational_emissions:
    emission_factors_kg_co2e_per_mwh:
      nuclear: 0
      gas: 370
      coal: 820
      oil: 650
      wind: 0
      solar: 0
      hydro: 0
      bioenergy: 230
""",
        encoding="utf-8",
    )

    result = run_marginal_emissions_proxy(
        input_path=input_path,
        emission_factors_path=factors_path,
        output_path=output_path,
        metrics_path=metrics_path,
    )

    assert result["rows"] == 6
    assert len(pd.read_csv(output_path)) == 6
    json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "NaN" not in metrics_path.read_text(encoding="utf-8")


def sample_hourly_carbon() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h"),
            "methodology": ["direct_operational_emissions"] * 3,
            "window": ["latest_test"] * 3,
            "model": ["model_a"] * 3,
            "actual_carbon_intensity_g_co2e_per_kwh": [100.0, 120.0, 140.0],
            "predicted_carbon_intensity_g_co2e_per_kwh": [90.0, 110.0, 130.0],
        }
    )
    actual_generation = {
        "nuclear": [100.0, 100.0, 100.0],
        "gas": [10.0, 20.0, 20.0],
        "coal": [0.0, 0.0, 5.0],
        "oil": [0.0, 0.0, 0.0],
        "wind": [30.0, 25.0, 25.0],
        "solar": [5.0, 5.0, 5.0],
        "hydro": [20.0, 20.0, 20.0],
        "bioenergy": [5.0, 5.0, 10.0],
    }
    for source, values in actual_generation.items():
        frame[f"actual_{source}_generation_mwh"] = values
        frame[f"predicted_{source}_generation_mwh"] = [value + 1.0 for value in values]
    return frame
