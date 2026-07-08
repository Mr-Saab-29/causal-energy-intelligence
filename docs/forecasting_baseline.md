# Forecasting Baseline

This baseline forecasts France day-ahead electricity spot prices using the stable modeling dataset.

## Target

- `price_eur_mwh`

## Strict Forecasting Features

Only time-derived and lagged/rolling features are used:

- `hour`, `day_of_week`, `month`, `day_of_year`
- `is_weekend`, `is_peak_hour`
- `hour_sin`, `hour_cos`
- `day_of_year_sin`, `day_of_year_cos`
- `price_lag_1h`
- `price_lag_24h`
- `price_lag_168h`
- `price_rolling_mean_24h`
- `price_rolling_mean_168h`
- `consumption_lag_24h`
- `wind_lag_24h`
- `solar_lag_24h`

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

- MAE
- RMSE
- sMAPE
- directional accuracy

The pipeline also writes error diagnostics by:

- hour of day
- day of week
- actual price regime

## Run

From the repository root:

```bash
python3 - <<'PY'
from src.models.baseline_price import run_price_baselines

result = run_price_baselines()
print(result["summary"])
PY
```

## Outputs

- Metrics: `reports/metrics/price_baseline_metrics.json`
- Predictions: `reports/predictions/price_baseline_predictions.csv`
- Error diagnostics: `reports/metrics/price_baseline_error_diagnostics.csv`
- Model artifacts:
  - `models/ridge_price_baseline.joblib`
  - `models/random_forest_price_baseline.joblib`
  - `models/hist_gradient_boosting_price_baseline.joblib`
  - `models/lightgbm_price_baseline.joblib`
  - `models/xgboost_price_baseline.joblib`
