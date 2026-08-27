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

Build the Sprint 2 marginal-emissions proxy from existing hourly carbon outputs:

```bash
make marginal-emissions
```

Compare average-carbon rankings against marginal-carbon rankings and export
causal-adjusted recommendations:

```bash
make causal-recommendations
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
- `reports/carbon/marginal_emissions_proxy.csv`
  - hourly marginal-emissions proxy for actual and predicted source-generation bases
  - positive generation response by source
  - marginal source, source response shares, marginal intensity, and average-vs-marginal delta
- `reports/metrics/marginal_emissions_proxy_metrics.json`
  - coverage, mean/median marginal intensity, average-vs-marginal delta, and marginal-source mix
- `reports/rankings/marginal_workload_decision_rankings.csv`
  - the existing workload candidate table re-ranked with marginal-carbon intensity
  - preserves average-carbon ranks in `average_*` columns
- `reports/recommendations/causal_adjusted_workload_recommendations.csv`
  - top-5 recommendations from marginal-carbon rankings with the existing uncertainty guard and confidence calibration
- `reports/metrics/marginal_ranking_shift_metrics.json`
  - top-1 change rate, top-5 overlap, rank displacement, causal coverage, and regret deltas

Carbon intensity uses:

```text
carbon_intensity_g_co2e_per_kwh = total_emissions_kg_co2e / total_generation_mwh
```

This works because `1 kg CO2e/MWh = 1 g CO2e/kWh`.

The marginal-emissions proxy uses hour-to-hour source-level generation changes.
For each hour, positive generation deltas are treated as the responding marginal
mix and weighted by the configured emission factors:

```text
marginal_intensity_g_co2e_per_kwh =
  sum(max(delta_source_mwh, 0) * source_emission_factor_kg_co2e_per_mwh)
  / sum(max(delta_source_mwh, 0))
```

Hours with no positive generation response have unavailable marginal intensity.
This is a proxy, not a structural dispatch model; it is intended for ranking
comparison before causal-adjusted recommendations become the champion path.
