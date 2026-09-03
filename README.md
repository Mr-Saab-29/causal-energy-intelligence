# Causal Energy Intelligence Platform

Clean-hour scheduling, decision ranking, and future causal analysis for carbon-aware workload shifting.

## Architecture

```text
External APIs / CSV sources
  -> Local ingestion + Dagster refresh skeleton
  -> Supabase/Postgres transformed tables or local processed files
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

Vercel deployment:

- Set the Vercel project root directory to `frontend`.
- Use build command `npm run build`.
- Use output directory `dist`.
- Automatic Vercel Git deployments are disabled in `frontend/vercel.json`. Deploy through the scheduled GitHub Actions workflow so the generated live dashboard data is included.
- The local/frontend build still includes a sample dashboard data contract if `frontend/public/data/dashboard.json` has not been generated. That keeps local builds healthy, but it is not the production deployment path.

Daily automated deployment uses GitHub Actions and Vercel CLI so generated
dashboard data does not need to be committed. Add these GitHub repository
secrets before enabling the scheduled deployment:

- `SUPABASE_DATABASE_URL`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

The scheduled workflow ingests data, monitors drift, retrains through the gated
promotion path only when needed or when model artifacts are missing, rebuilds
future recommendations, writes `frontend/public/data/dashboard.json`, and
deploys the prebuilt frontend to Vercel.

Full retraining and dashboard refresh:

```bash
make forecast-all
```

This runs a gated retrain. The current production artifacts are snapshotted, a candidate model set is
trained and evaluated, and the candidate is promoted only if it beats the incumbent under the
carbon-first decision-support score. If the candidate is worse, the previous model/report/dashboard
artifacts are restored.

## Current Status

The platform now has a working France electricity decision-support baseline:

- Canonical ETL contracts and Supabase/Postgres schemas are in place.
- Local ingestion can refresh France day-ahead spot prices, electricity mix, production, consumption, and weather-derived modeling features through the latest available date.
- Scheduled cloud ingestion now writes transformed canonical rows to Supabase instead of relying on persisted raw API payloads or cached source CSVs. To stay inside the Supabase free tier, scheduled Supabase ingestion loads national `FR` electricity mix by default, skips regional mix rows, and exports the modeling feature CSV as a temporary runner artifact instead of storing derived feature rows in Supabase.
- Future exogenous weather ingestion is available for the next 24 hours and can upsert transformed forecast rows to Supabase.
- The modeling dataset is built at `data/processed/modeling_price_features.csv`.
- Price models are treated as supporting signals only; the project is no longer framed around point spot-price prediction.
- The primary decision output is a top-5 list of recommended clean workload start hours from the combined scheduling ranking.
- Operational recommendations are written to `reports/recommendations/future_champion_workload_recommendations.csv`.
- Future scenario recommendations are written to `reports/scenarios/future_workload_scenario_recommendations.csv`.
- Recommendations show price direction versus the previous day at the same time instead of presenting price as the main dashboard forecast.
- The champion model is selected from generated metrics with a regret-first score: 35% realized recommendation regret, 25% carbon regret, 20% top-5 ranking loss, 10% price-direction error, and 10% carbon-intensity error.
- Full retraining is guarded by an incumbent-vs-candidate promotion gate. A candidate retrain is promoted only when its weighted lower-is-better decision metrics beat the current production champion and recommendation/carbon regret do not regress beyond tolerance; otherwise the incumbent artifacts are restored.
- The latest promotion decision is written to `reports/metrics/model_promotion_decision.json`.
- Historical policy backtests evaluate the exact emitted rank-1 recommendation by model and scenario.
- Scenario-level champion selection reports the best model separately for clean-first, balanced, and cost-aware-clean preferences.
- The ranking layer is evaluated by top-k capture, pairwise ranking loss by decision day/window, top-5 classification metrics, regret by day/window, and savings versus running immediately.
- A ranking-specific top-5 classifier is trained on historical decision candidates and accepted only when out-of-window combined regret and carbon regret do not degrade versus the baseline ranking score.
- Candidate hours with weak raw price/carbon score separation receive an uncertainty penalty before recommendation ranking. When no low-uncertainty candidate exists, the export marks the row as `no_low_risk_recommendation_available`.
- Empirical prediction-interval half-widths are calibrated from historical candidate residual quantiles and reused for future recommendation uncertainty.
- Workload recommendations support duration, earliest start, latest end, max-delay, price-weight, and carbon-weight constraints.
- Scenario reranking is available for clean-first, balanced, and cost-aware-clean preferences. The dashboard scenario selector now changes the active future top-5 recommendation list, KPIs, and chart.
- Ranking currently uses strict forecast-time features: calendar features, lagged prices, lagged/rolling supply-demand signals, and upstream forecasted consumption/production.
- Upstream baselines forecast consumption, total production, and source-level production for nuclear, gas, coal, oil, wind, solar, hydro, and bioenergy.
- Forecast diagnostics include MAE, RMSE, sMAPE, directional accuracy, top-error periods, grouped error diagnostics, ranking metrics, regret metrics, and feature importance.
- Historical validation windows are assigned dynamically from the ingested data. The final validation/test window is the latest 90 days ending at the latest modeling timestamp.
- The dashboard shows a health/status band from `reports/metrics/pipeline_health.json`.
- The dashboard shows forecast-monitoring status from `reports/metrics/forecast_monitoring.json`, including whether retraining is recommended and why. It also marks the monitor as stale if model-quality artifacts changed after the monitor report was generated.
- Recommendation confidence is calibrated from historical confidence bins, empirical top-5 hit rates, and observed regret, with minimum sample-size guards and scenario-specific calibration for scenario rerankings.
- The dashboard recommendation rows now show explicit carbon intensity, price direction versus yesterday, calibrated confidence, expected regret, and expandable details.
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
make train-all-gated
make forecast-all
make forecast-all-force
make operational-refresh
make operational-publish
make ingest-monitor
make ingest-monitor-cloud
make forecast-monitor
make pipeline-health
make dagster-dev
```

Command intent:

- `make train-all` retrains historical models and validation artifacts only.
- `make train-all-gated` runs the historical retrain behind the incumbent promotion gate and is used by the Dagster full-refresh asset.
- `make operational-refresh` uses current saved model artifacts to build next-24-hour future recommendations, future scenario recommendations, health/monitor reports, and the dashboard.
- `make operational-publish` runs the fast publish stage after data/model state already exists: recommendations, health/monitor reports, dashboard data, and frontend build.
- `make forecast-all` runs the gated full retrain and promotes the candidate only if it beats the incumbent. This is the default safe retraining command.
- `make forecast-all-candidate` is the internal ungated candidate pipeline used by the promotion gate.
- `make forecast-all-force` retrains and overwrites artifacts without the incumbent promotion gate. Use only when you intentionally want to bypass the guard.
- `make ingest-monitor` ingests latest API data, runs pipeline health, and writes the forecast monitoring report without retraining.
- `make ingest-monitor-cloud` is the deployable scheduled variant. It requires `DATABASE_URL`, limits historical ingestion to a recent 14-day lookback, writes transformed rows to Supabase, refreshes future weather, and avoids expensive bootstrap backfills.
- `make forecast-monitor` writes `reports/metrics/forecast_monitoring.json` from existing artifacts.
- `make dagster-dev` starts the local Dagster UI. The main jobs are `ingestion_monitor_refresh`, `daily_clean_hour_refresh`, and `quick_recommendation_refresh`.
- Dagster schedules `ingestion_monitor_refresh` for `02:00` Europe/Paris every day. This scheduled job does not retrain models.
- GitHub Actions workflow `.github/workflows/daily-ingestion-monitor.yml` provides a deployable no-cost daily ingestion monitor. It runs ingestion, preflight monitoring, optional gated retraining, and dashboard publishing as separate chained jobs so failed downstream jobs can be rerun without repeating a completed retrain.
- Monitoring trigger thresholds live in `config/monitoring_thresholds.yaml`.

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
- Ranking model metrics: `reports/metrics/ranking_model_metrics.json`
- Confidence calibration: `reports/metrics/recommendation_confidence_calibration.json`
- Scenario confidence calibration: `reports/metrics/scenario_recommendation_confidence_calibration.json`
- Historical recommendation drift metrics: `reports/metrics/recommendation_drift_metrics.json`
- Prediction interval calibration: `reports/metrics/recommendation_prediction_interval_calibration.json`
- Recommendation policy backtest: `reports/metrics/recommendation_policy_backtest.json`
- Scenario champion selection: `reports/metrics/scenario_champion_selection.json`
- Monitoring thresholds: `config/monitoring_thresholds.yaml`
- Champion model selection: `reports/metrics/champion_model_selection.json`
- Scenario rerankings: `reports/scenarios/workload_scenario_recommendations.csv`
- Future scenario rerankings: `reports/scenarios/future_workload_scenario_recommendations.csv`
- Scenario metrics: `reports/metrics/scenario_reranking_metrics.json`
- Future weather forecast: `data/processed/future_weather_forecast.csv`
- Future decision rankings: `reports/rankings/future_workload_decision_rankings.csv`
- Future champion recommendations: `reports/recommendations/future_champion_workload_recommendations.csv`
- Future recommendation metadata: `reports/metrics/future_recommendation_metadata.json`
- Recommendation drift metrics: `reports/metrics/future_recommendation_drift_metrics.json`
- Pipeline health: `reports/metrics/pipeline_health.json`
- Forecast monitoring: `reports/metrics/forecast_monitoring.json`
- Model promotion decision: `reports/metrics/model_promotion_decision.json`
- Operational forecast history: `reports/monitoring/operational_ranking_history.csv`
- Dashboard data contract: `frontend/public/data/dashboard.json`
- Supply/demand metrics: `reports/metrics/supply_demand_baseline_metrics.json`
- Supply/demand predictions: `reports/predictions/supply_demand_baseline_predictions.csv`
- Feature importance: `reports/metrics/*feature_importance.csv`

## To Do Next

Status: in progress.

- Continue improving the ranking-specific model until it clears the guarded acceptance gate consistently.
- Extend uncertainty calibration beyond confidence bins with prediction intervals or conformal-style bands.
- Tune the model promotion gate using more stable out-of-time windows and operational settled-forecast metrics once enough daily history accumulates.
- Move the deployable daily orchestration from monitor-only refresh toward artifact publishing for the dashboard.
- Decide whether Docker should stay optional or be repaired for a full local compose workflow.
- Add a live API layer after the static dashboard contract stabilizes.
- Start causal modeling after the deployment and refresh path is reliable.
- Expand workload constraints for real operational use cases, such as multi-hour jobs, deadlines, blackout windows, and regional constraints.

## Data Contracts

Canonical contracts are defined in `src/data/contracts.py` and documented in `docs/data_contracts.md`. Apply the Supabase/Postgres setup from `db/schema.sql`, `db/feature_views.sql`, and `db/modeling_features.sql`.

Source-specific extraction notes are documented in `docs/data_sources.md`.

France electricity spot-price provider tradeoffs are documented in `docs/price_source_options.md`. The default spot-price source is Energy-Charts, with ENTSO-E kept as a fallback.

Supabase loading instructions are documented in `docs/supabase_load.md`.

Post-ingestion validation is documented in `docs/data_validation.md`.

Modeling dataset construction is documented in `docs/modeling_dataset.md`.

Forecasting baselines are documented in `docs/forecasting_baseline.md`.

MLOps, MLflow tracking, Docker Compose, and Dagster refresh notes are documented in
`docs/mlops_orchestration.md`.
