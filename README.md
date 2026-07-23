# Causal Energy Intelligence Platform

Clean-hour scheduling, causal inference, and what-if optimization for carbon-aware workload shifting.

## Architecture

```text
External APIs / CSV sources
  -> Airflow ETL
  -> Cloud Postgres
  -> Feature Engineering
  -> Clean-Hour Forecasting + Ranking Models
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
- `src/models/` — Forecast training, evaluation, and clean-hour scoring signals.
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

## Clean-Hour Dashboard

The frontend lives in `frontend/` and is Vercel-ready. It reads the generated static data contract at
`frontend/public/data/dashboard.json`, which can later be replaced by a live API once Airflow refreshes
the source data daily.

```bash
make forecast-recommendations
make frontend-install
make frontend-dev
```

Production build:

```bash
make frontend-build
```

## Current Status

The platform now has a working France electricity decision-support baseline:

- Canonical ETL contracts and Supabase/Postgres schemas are in place.
- France day-ahead spot prices, electricity mix, production, consumption, and weather-derived modeling features are supported.
- The modeling dataset is built at `data/processed/modeling_price_features.csv`.
- Price models are treated as supporting signals only; the project is no longer framed around point spot-price prediction.
- The primary decision output is a top-5 list of recommended clean workload start hours from the combined scheduling ranking.
- Recommendations show price direction versus the previous day at the same time instead of presenting price as the main dashboard forecast.
- The champion model is selected from generated metrics with a carbon-first score: 45% carbon-intensity error, 25% carbon regret, 20% top-5 ranking loss, and 10% price-direction error.
- The ranking layer is evaluated by top-k capture, pairwise ranking loss, top-5 classification metrics, regret by day/window, and savings versus running immediately.
- Workload recommendations support duration, earliest start, latest end, max-delay, price-weight, and carbon-weight constraints.
- Scenario reranking is available for clean-first, balanced, and cost-aware-clean preferences.
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
make forecast-scenarios
make forecast-all
```

Current key artifacts:

- Price metrics: `reports/metrics/price_baseline_metrics.json`
- Price predictions: `reports/predictions/price_baseline_predictions.csv`
- Decision rankings: `reports/rankings/price_decision_rankings.csv`
- Ranking metrics: `reports/metrics/price_ranking_metrics.json`
- Combined workload rankings: `reports/rankings/workload_decision_rankings.csv`
- Top 5 workload recommendations: `reports/recommendations/top5_workload_recommendations.csv`
- Champion-only recommendations: `reports/recommendations/champion_workload_recommendations.csv`
- Combined workload metrics: `reports/metrics/workload_decision_metrics.json`
- Ranking-specific metrics: `reports/metrics/ranking_specific_metrics.json`
- Champion model selection: `reports/metrics/champion_model_selection.json`
- Scenario rerankings: `reports/scenarios/workload_scenario_recommendations.csv`
- Scenario metrics: `reports/metrics/scenario_reranking_metrics.json`
- Dashboard data contract: `frontend/public/data/dashboard.json`
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
