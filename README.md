# Causal Energy Intelligence Platform

Forecasting, causal inference, and what-if optimization for ranked energy decisions and carbon-aware workload shifting.

## Architecture

```text
External APIs / CSV sources
  -> Airflow ETL
  -> Cloud Postgres
  -> Feature Engineering
  -> Forecasting + Ranking Models
  -> Causal Inference Engine
  -> What-if Simulator
  -> FastAPI
  -> Docker
  -> Monitoring
  -> Optional local Kubernetes
```

## Repository Layout

- `dags/` — Airflow ETL orchestration.
- `src/data/` — Extract, transform, and load utilities.
- `src/features/` — Feature engineering.
- `src/models/` — Forecast training, evaluation, and prediction.
- `src/causal/` — Causal DAGs, effect estimation, and counterfactuals.
- `src/optimization/` — Carbon-aware workload shifting.
- `src/monitoring/` — Metrics and observability helpers.
- `api/` — FastAPI application.
- `notebooks/` — EDA, forecasting, and causal analysis notebooks.
- `docker/` — Container and local compose setup.
- `k8s/` — Optional local Kubernetes manifests.
- `docs/` — Architecture, causal DAG, and project report notes.
- `db/` — Supabase/Postgres schema.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Current Status

The platform now has a working France electricity decision-support baseline:

- Canonical ETL contracts and Supabase/Postgres schemas are in place.
- France day-ahead spot prices, electricity mix, production, consumption, and weather-derived modeling features are supported.
- The modeling dataset is built at `data/processed/modeling_price_features.csv`.
- Price models are treated as internal scoring signals rather than the final objective.
- The primary decision output is a top-5 list of recommended workload start hours from the combined price/carbon ranking.
- The ranking layer is evaluated by top-k capture, rank correlation, regret versus the actual best hour, and savings versus running immediately.
- Workload recommendations support duration, earliest start, latest end, max-delay, price-weight, and carbon-weight constraints.
- Ranking currently uses strict forecast-time features: calendar features, lagged prices, lagged/rolling supply-demand signals, and upstream forecasted consumption/production.
- Upstream baselines forecast consumption, total production, and source-level production for nuclear, gas, coal, oil, wind, solar, hydro, and bioenergy.
- Forecast diagnostics include MAE, RMSE, sMAPE, directional accuracy, top-error periods, grouped error diagnostics, ranking metrics, regret metrics, and feature importance.
- Notebook `notebooks/02_forecasting.ipynb` reads the generated metrics and diagnostics.

Common forecast commands:

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

Current key artifacts:

- Price metrics: `reports/metrics/price_baseline_metrics.json`
- Price predictions: `reports/predictions/price_baseline_predictions.csv`
- Decision rankings: `reports/rankings/price_decision_rankings.csv`
- Ranking metrics: `reports/metrics/price_ranking_metrics.json`
- Combined workload rankings: `reports/rankings/workload_decision_rankings.csv`
- Top 5 workload recommendations: `reports/recommendations/top5_workload_recommendations.csv`
- Combined workload metrics: `reports/metrics/workload_decision_metrics.json`
- Supply/demand metrics: `reports/metrics/supply_demand_baseline_metrics.json`
- Supply/demand predictions: `reports/predictions/supply_demand_baseline_predictions.csv`
- Feature importance: `reports/metrics/*feature_importance.csv`

Remaining work includes productionizing the API, improving ranking models, adding scenario-based reranking, causal effect estimation, and richer workload-shifting constraints.

## Data Contracts

Canonical contracts are defined in `src/data/contracts.py` and documented in `docs/data_contracts.md`. Apply the initial Supabase/Postgres schema from `db/schema.sql`.

Source-specific extraction notes are documented in `docs/data_sources.md`.

France electricity spot-price provider tradeoffs are documented in `docs/price_source_options.md`. The default spot-price source is Energy-Charts, with ENTSO-E kept as a fallback.

Supabase loading instructions are documented in `docs/supabase_load.md`.

Post-ingestion validation is documented in `docs/data_validation.md`.

Modeling dataset construction is documented in `docs/modeling_dataset.md`.

Forecasting baselines are documented in `docs/forecasting_baseline.md`.
