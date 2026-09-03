# Carbon Accounting

Power-sector emissions and carbon intensity are calculated from source-level hourly generation forecasts and actuals.

## Configuration

Emission factors live in `config/emission_factors.yaml`.

The config supports multiple accounting methodologies under `methodologies`, each with source-level factors in `kg_co2e_per_mwh`:

- `direct_operational_emissions`
- `lifecycle_emissions`

Do not add emission factors inside calculation code. Update or add methodologies in the YAML file instead.

## Run

Generate source-level production forecasts and carbon accounting outputs:

```bash
python -m src.models.train_forecast --target production
```

Recompute carbon outputs from an existing `reports/predictions/production_baseline_predictions.csv` file:

```bash
python -m src.models.train_forecast --target carbon
```

## Outputs

- `reports/carbon/hourly_carbon_intensity.csv`
  - actual hourly power-sector emissions
  - predicted hourly power-sector emissions
  - actual hourly carbon intensity
  - predicted hourly carbon intensity
  - source-level generation and emission columns
- `reports/carbon/technology_emission_contributions.csv`
  - long-form technology-level emission contributions and shares
- `reports/metrics/carbon_forecast_metrics.json`
  - emissions and carbon-intensity MAE, RMSE, bias, and sMAPE by methodology/window/model

Carbon intensity uses:

```text
carbon_intensity_g_co2e_per_kwh = total_emissions_kg_co2e / total_generation_mwh
```

This works because `1 kg CO2e/MWh = 1 g CO2e/kWh`.

## Marginal Emissions Proxy MVP

Sprint 2 uses a proxy, not a full grid-dispatch causal estimate. The operational proxy derives implied generation from total emissions and average carbon intensity, then estimates marginal intensity from positive hour-to-hour changes:

```text
implied_generation_mwh = total_emissions_kg_co2e / average_carbon_intensity_g_co2e_per_kwh
marginal_proxy = positive_delta(total_emissions_kg_co2e) / positive_delta(implied_generation_mwh)
```

Rows without a usable positive emissions and generation delta fall back to average carbon intensity and are labelled `average_carbon_fallback`. Rows with a usable proxy are labelled `marginal_emissions_proxy`.

Run:

```bash
make causal-recommendations
```

Primary outputs:

- `reports/rankings/future_marginal_workload_decision_rankings.csv`
- `reports/recommendations/future_causal_adjusted_workload_recommendations.csv`
- `reports/metrics/marginal_ranking_shift_metrics.json`
