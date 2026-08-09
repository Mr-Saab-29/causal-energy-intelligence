# Causal Energy Intelligence Platform

Clean-hour scheduling, decision ranking, and future causal analysis for carbon-aware workload shifting.

## Architecture

```text
External APIs / CSV sources
  -> Local ingestion + Dagster refresh skeleton
  -> Supabase/Postgres or local processed files
  -> Feature Engineering
  -> Consumption, Production, Source Production Models
  -> Carbon Intensity Estimates
  -> Clean-Hour Decision Ranking
  -> Next-24h Future Recommendations
  -> Static Dashboard Data Contract
  -> Vercel-ready Frontend
  -> MLflow + Dagster Observability
  -> Causal Inference Engine (in progress)
```

## Repository Layout

- `orchestration/` — Dagster daily and quick refresh assets.
- `src/data/` — Extract, transform, and load utilities.
- `src/features/` — Feature engineering.
- `src/models/` — Forecast training, evaluation, saved artifacts, and future recommendation scoring.
- `src/causal/` — Causal DAGs, effect estimation, and counterfactuals.
- `src/optimization/` — Carbon-aware workload shifting.
- `src/monitoring/` — Metrics and observability helpers.
- `api/` — FastAPI application.
- `frontend/` — Vercel-ready clean-hour decision dashboard.
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
`frontend/public/data/dashboard.json`, which can later be replaced by a live API. The dashboard now
prefers the operational next-24-hour recommendation artifact when it exists, so it is meant to show
future scheduling decisions rather than historical validation rows.

```bash
make frontend-install
make operational-refresh
make frontend-dev
```

Production build:

```bash
make frontend-build
```

Full retraining and dashboard refresh:

```bash
make forecast-all
```

This retrains the historical models, selects/saves champion artifacts, fetches next-24-hour future
exogenous weather, builds future recommendations, rebuilds the dashboard data contract, and builds
the frontend.

## Current Status

The platform now has a working France electricity decision-support baseline:

- Canonical ETL contracts and Supabase/Postgres schemas are in place.
- Local ingestion can refresh France day-ahead spot prices, electricity mix, production, consumption, and weather-derived modeling features through the latest available date.
- Future exogenous weather ingestion is available for the next 24 hours.
- The modeling dataset is built at `data/processed/modeling_price_features.csv`.
- Price models are treated as supporting signals only; the project is no longer framed around point spot-price prediction.
- The primary decision output is a top-5 list of recommended clean workload start hours from the combined scheduling ranking.
- Operational recommendations are written to `reports/recommendations/future_champion_workload_recommendations.csv`.
- Recommendations show price direction versus the previous day at the same time instead of presenting price as the main dashboard forecast.
- The champion model is selected from generated metrics with a carbon-first score: 45% carbon-intensity error, 25% carbon regret, 20% top-5 ranking loss, and 10% price-direction error.
- The ranking layer is evaluated by top-k capture, pairwise ranking loss, top-5 classification metrics, regret by day/window, and savings versus running immediately.
- Workload recommendations support duration, earliest start, latest end, max-delay, price-weight, and carbon-weight constraints.
- Scenario reranking is available for clean-first, balanced, and cost-aware-clean preferences.
- Ranking currently uses strict forecast-time features: calendar features, lagged prices, lagged/rolling supply-demand signals, and upstream forecasted consumption/production.
- Upstream baselines forecast consumption, total production, and source-level production for nuclear, gas, coal, oil, wind, solar, hydro, and bioenergy.
- Forecast diagnostics include MAE, RMSE, sMAPE, directional accuracy, top-error periods, grouped error diagnostics, ranking metrics, regret metrics, and feature importance.
- Historical validation windows are assigned dynamically from the ingested data. The final validation/test window is the latest 90 days ending at the latest modeling timestamp.
- The dashboard shows a health/status band from `reports/metrics/pipeline_health.json`.
- The dashboard recommendation rows now show explicit carbon intensity, price direction versus yesterday, confidence, and expandable details.
- MLflow tracking hooks and a Dagster refresh skeleton are in place. Docker files exist, but Docker execution is currently optional and can be skipped during local development.
- Notebook `notebooks/02_forecasting.ipynb` reads the generated metrics and diagnostics.

Common commands:

```bash
make ingest-plan
make ingest-latest
make ingest-future
make forecast-consumption
make forecast-production
make forecast-supply-demand
make forecast-price
make forecast-ranking
make forecast-decision
make forecast-recommendations
make forecast-scenarios
make train-all
make forecast-all
make operational-refresh
make pipeline-health
make dagster-dev
```

Command intent:

- `make train-all` retrains historical models and validation artifacts only.
- `make operational-refresh` uses current saved model artifacts to build next-24-hour future recommendations and the dashboard.
- `make forecast-all` retrains everything, rebuilds future recommendations, and builds the dashboard/frontend.
- `make dagster-dev` starts the local Dagster UI. The main jobs are `daily_clean_hour_refresh` and `quick_recommendation_refresh`.

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
- Future weather forecast: `data/processed/future_weather_forecast.csv`
- Future decision rankings: `reports/rankings/future_workload_decision_rankings.csv`
- Future champion recommendations: `reports/recommendations/future_champion_workload_recommendations.csv`
- Future recommendation metadata: `reports/metrics/future_recommendation_metadata.json`
- Pipeline health: `reports/metrics/pipeline_health.json`
- Dashboard data contract: `frontend/public/data/dashboard.json`
- Supply/demand metrics: `reports/metrics/supply_demand_baseline_metrics.json`
- Supply/demand predictions: `reports/predictions/supply_demand_baseline_predictions.csv`
- Feature importance: `reports/metrics/*feature_importance.csv`

## To Do Next

Status: in progress.

- Improve the ranking-specific modeling layer beyond baseline regression-derived scores.
- Add stronger uncertainty and confidence calibration for operational recommendations.
- Add real data freshness checks for future exogenous data, not only historical source files.
- Move from local Dagster skeleton to a deployable daily orchestration setup.
- Decide whether Docker should stay optional or be repaired for a full local compose workflow.
- Add a live API layer after the static dashboard contract stabilizes.
- Start causal modeling after the deployment and refresh path is reliable.
- Expand workload constraints for real operational use cases, such as multi-hour jobs, deadlines, blackout windows, and regional constraints.

## Data Contracts

Canonical contracts are defined in `src/data/contracts.py` and documented in `docs/data_contracts.md`. Apply the initial Supabase/Postgres schema from `db/schema.sql`.

Source-specific extraction notes are documented in `docs/data_sources.md`.

France electricity spot-price provider tradeoffs are documented in `docs/price_source_options.md`. The default spot-price source is Energy-Charts, with ENTSO-E kept as a fallback.

Supabase loading instructions are documented in `docs/supabase_load.md`.

Post-ingestion validation is documented in `docs/data_validation.md`.

Modeling dataset construction is documented in `docs/modeling_dataset.md`.

Forecasting baselines are documented in `docs/forecasting_baseline.md`.

MLOps, MLflow tracking, Docker Compose, and Dagster refresh notes are documented in
`docs/mlops_orchestration.md`.
