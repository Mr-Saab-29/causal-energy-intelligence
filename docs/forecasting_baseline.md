# Forecasting Baseline

This baseline ranks candidate workload hours using France day-ahead electricity price signals from
the stable modeling dataset. Exact spot-price prediction is kept as an internal signal, not the final
decision objective.
It also trains upstream consumption and total-production forecasters, then feeds those forecasted
supply/demand signals into the price models.
Separate source-level generation forecasters are trained for nuclear, gas, coal, oil, wind, solar,
hydro, and bioenergy so those forecasts can feed later carbon-footprint estimates.

## Decision Target

- rank candidate hours from cheapest to most expensive within each decision day
- minimize regret versus the actual cheapest hour
- capture the actual cheapest hour in the model's top-k recommendations

## Strict Forecasting Features

Only time-derived and lagged/rolling features are used:

- `hour`, `day_of_week`, `month`, `day_of_year`
- `is_weekend`, `is_peak_hour`
- `is_morning_ramp`, `is_evening_peak`, `is_overnight`
- `hour_sin`, `hour_cos`
- `day_of_year_sin`, `day_of_year_cos`
- `price_lag_1h`
- `price_lag_24h`
- `price_lag_48h`
- `price_lag_72h`
- `price_lag_168h`
- `price_lag_336h`
- `price_rolling_mean_24h`
- `price_rolling_mean_168h`
- `price_rolling_std_24h`
- `price_rolling_min_24h`
- `price_rolling_max_24h`
- `price_rolling_range_24h`
- price momentum/spread features from lagged prices
- `consumption_lag_24h`
- `consumption_lag_168h`
- `wind_lag_24h`
- `wind_lag_168h`
- `solar_lag_24h`
- `solar_lag_168h`
- lagged residual demand and variable-renewable features
- forecasted consumption, total production, residual demand, and supply-demand gap

The contemporaneous electricity/weather columns remain in the dataset for causal analysis, but they are not used by the strict baseline forecasters.

## Walk-Forward Windows

Expanding-window validation:

| Window | Train End | Test Period |
| --- | --- | --- |
| `wf_2026_01` | 2025-12-31 | 2026-01 |
| `wf_2026_02` | 2026-01-31 | 2026-02 |
| `wf_2026_03` | 2026-02-28 | 2026-03 |
| `test_2026_q2` | 2026-03-31 | 2026-04-01 to 2026-06-30 |

## Models

- `naive_lag_24h`
- `ridge`
- `random_forest`
- `hist_gradient_boosting`
- `lightgbm`
- `xgboost`

## Metrics

Ranking metrics:

- top-1 hit rate
- top-3 capture rate
- mean/median top-1 regret in EUR/MWh
- mean actual rank of the predicted-best hour
- Spearman rank correlation

Point-forecast diagnostics are still retained as supporting signals:

- MAE
- RMSE
- sMAPE
- directional accuracy

The pipeline also writes error diagnostics by:

- walk-forward window
- hour of day
- day of week
- month
- weekend flag
- peak-hour flag
- morning ramp flag
- evening peak flag
- actual price regime

Diagnostics include raw-error fields (`mae`, `rmse`, `mean_error`, `max_abs_error`) and
relative-error fields (`smape`, `max_smape`). The top-error artifact also includes row-level
`smape_pct`, which is `100 * sMAPE`.

Current aggregate price performance after adding forecasted consumption and production signals:

| Model | MAE | RMSE | sMAPE | Directional Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `hist_gradient_boosting` | 8.198 | 12.917 | 0.352 | 0.755 |
| `lightgbm` | 8.229 | 12.931 | 0.352 | 0.761 |
| `xgboost` | 8.356 | 12.942 | 0.353 | 0.760 |
| `random_forest` | 8.900 | 13.989 | 0.365 | 0.745 |
| `ridge` | 10.512 | 14.753 | 0.407 | 0.727 |
| `naive_lag_24h` | 25.481 | 35.838 | 0.661 | 0.782 |

Current aggregate ranking performance:

| Model | Top-1 Hit | Top-3 Capture | Mean Top-1 Regret | Spearman |
| --- | ---: | ---: | ---: | ---: |
| `random_forest` | 0.306 | 0.769 | 3.341 EUR/MWh | 0.906 |
| `xgboost` | 0.300 | 0.769 | 3.443 EUR/MWh | 0.915 |
| `lightgbm` | 0.288 | 0.769 | 3.506 EUR/MWh | 0.915 |
| `hist_gradient_boosting` | 0.281 | 0.769 | 4.032 EUR/MWh | 0.915 |
| `ridge` | 0.188 | 0.688 | 4.762 EUR/MWh | 0.883 |
| `naive_lag_24h` | 0.344 | 0.663 | 8.237 EUR/MWh | 0.745 |

Current aggregate combined price/carbon decision performance with equal weights:

| Model | Top-1 Hit | Top-3 Capture | Cost Savings vs Run Now | Carbon Savings vs Run Now |
| --- | ---: | ---: | ---: | ---: |
| `hist_gradient_boosting` | 0.319 | 0.725 | 40.372 EUR/MWh | 3.643 gCO2e/kWh |
| `lightgbm` | 0.319 | 0.725 | 40.683 EUR/MWh | 3.638 gCO2e/kWh |
| `random_forest` | 0.206 | 0.725 | 39.890 EUR/MWh | 3.519 gCO2e/kWh |
| `xgboost` | 0.256 | 0.769 | 41.122 EUR/MWh | 3.628 gCO2e/kWh |
| `ridge` | 0.250 | 0.669 | 40.239 EUR/MWh | 3.610 gCO2e/kWh |
| `naive_lag_24h` | 0.194 | 0.438 | 32.325 EUR/MWh | 1.974 gCO2e/kWh |

Current aggregate upstream signal performance:

| Target | Best Model | MAE | RMSE | sMAPE | Directional Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| consumption | `ridge` | 658.820 | 1020.692 | 0.020 | 0.801 |
| production | `ridge` | 594.487 | 1025.238 | 0.015 | 0.793 |

## Current Error Findings

For the best MAE model, `hist_gradient_boosting`, the largest errors remain concentrated in two situations:

- Extreme negative prices around 2026-04-25 and 2026-04-26. The model predicts negative prices but underestimates the depth of the event, with absolute errors above 300 EUR/MWh at 2026-04-26 11:00-12:00 UTC.
- Evening price spikes around 17:00-18:00 UTC. The tree models tend to underpredict these high-price periods.

The feature-importance artifact shows that `price_lag_1h`, hour-of-day effects, `price_lag_24h`, and lagged price momentum features dominate the current tree models.

## Run

From the repository root:

```bash
make forecast-consumption
make forecast-production
make forecast-supply-demand
make forecast-price
make forecast-ranking
make forecast-decision
make forecast-recommendations
make forecast-all
```

`forecast-price` and `forecast-all` train upstream consumption/production forecasts first, then train
the price models with those forecasted values.
`forecast-ranking` rebuilds decision rankings from saved price predictions without retraining.
`forecast-decision` combines saved price rankings and carbon-intensity estimates into workload
recommendations. Optional constraints include `--duration-hours`, `--earliest-start-utc`,
`--latest-end-utc`, `--max-delay-hours`, `--price-weight`, and `--carbon-weight`.
`forecast-recommendations` exports the top 5 recommended workload start hours from the combined
decision ranking.
`forecast-production` trains total production plus the individual source-level production targets.

Equivalent direct Python commands:

```bash
python -m src.models.train_forecast --target consumption
python -m src.models.train_forecast --target production
python -m src.models.train_forecast --target supply-demand
python -m src.models.train_forecast --target price
python -m src.models.train_forecast --target ranking
python -m src.models.train_forecast --target decision
python -m src.models.train_forecast --target decision --top-n-recommendations 5
python -m src.models.train_forecast --target all
```

Example constrained workload ranking:

```bash
python -m src.models.train_forecast \
  --target decision \
  --duration-hours 3 \
  --earliest-start-utc 2026-04-01T08:00:00+00:00 \
  --latest-end-utc 2026-04-01T22:00:00+00:00 \
  --price-weight 0.5 \
  --carbon-weight 0.5
```

## Outputs

- Metrics: `reports/metrics/price_baseline_metrics.json`
- Predictions: `reports/predictions/price_baseline_predictions.csv`
- Error diagnostics: `reports/metrics/price_baseline_error_diagnostics.csv`
- Top forecast misses: `reports/metrics/price_baseline_top_errors.csv`
- Feature importance: `reports/metrics/price_baseline_feature_importance.csv`
- Decision rankings: `reports/rankings/price_decision_rankings.csv`
- Ranking metrics: `reports/metrics/price_ranking_metrics.json`
- Combined workload rankings: `reports/rankings/workload_decision_rankings.csv`
- Top 5 workload recommendations: `reports/recommendations/top5_workload_recommendations.csv`
- Combined workload metrics: `reports/metrics/workload_decision_metrics.json`
- Supply/demand metrics: `reports/metrics/supply_demand_baseline_metrics.json`
- Supply/demand predictions: `reports/predictions/supply_demand_baseline_predictions.csv`
- Supply/demand feature importance: `reports/metrics/supply_demand_baseline_feature_importance.csv`
- Consumption-only metrics: `reports/metrics/consumption_baseline_metrics.json`
- Consumption-only predictions: `reports/predictions/consumption_baseline_predictions.csv`
- Consumption-only feature importance: `reports/metrics/consumption_baseline_feature_importance.csv`
- Production-only metrics: `reports/metrics/production_baseline_metrics.json`
- Production-only predictions: `reports/predictions/production_baseline_predictions.csv`
- Production-only feature importance: `reports/metrics/production_baseline_feature_importance.csv`
- Production-source metrics from `forecast-all`: `reports/metrics/production_sources_baseline_metrics.json`
- Production-source predictions from `forecast-all`: `reports/predictions/production_sources_baseline_predictions.csv`
- Production-source feature importance from `forecast-all`: `reports/metrics/production_sources_baseline_feature_importance.csv`
- Model artifacts:
  - `models/ridge_price_baseline.joblib`
  - `models/random_forest_price_baseline.joblib`
  - `models/hist_gradient_boosting_price_baseline.joblib`
  - `models/lightgbm_price_baseline.joblib`
  - `models/xgboost_price_baseline.joblib`
