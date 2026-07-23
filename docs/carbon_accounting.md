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
