# Frozen t0-alpha benchmark — Day 1

## Energy aggregation correction — v3 workflow

**The v1/v2 model rankings are invalidated by a confirmed historical energy
aggregation bug.** Historical actuals have 30-minute intervals, not 15-minute
intervals. The earlier completeness checks below did not detect the unit error.
The corrected implementation is added; rebuilding and evaluating remain manual.

The default benchmark directory is now
`reports/benchmarks/t0_2026_05_26_to_08_23_v3`. Set `BENCHMARK_SNAPSHOT` before
starting Python to select another new directory. Existing v2 artifacts are
preserved. Training, validation, inference and report generation reject snapshots
without corrected-source evidence. The dashboard marks the old results as
requiring recomputation until a corrected report is published.

### What is rebuilt

`src.benchmarks.rebuild_energy` downloads national ODRE readings for the entire
saved source period, including pre-evaluation training history. It prefers a
complete consolidated/definitive hourly mix and falls back to real-time data
when that mix is unavailable. Every energy target is reconstructed; old target
values are never filled into the new source and there is no blanket doubling.
Only historical weather is retained from the previous recovery file.

The shared aggregator uses the explicit dataset cadence, counts duplicate UTC
instants once only when all consumed actual fields and source nature agree,
rejects conflicting duplicates or misaligned readings, leaves incomplete hourly fields missing, and requires all
generation components before computing total production. Two surviving samples
from a four-sample real-time hour are not mistaken for complete half-hour data.

Daily raw responses and hashes are cached in the adjacent
`t0_2026_05_26_to_08_23_v3_energy_rebuild/raw/` folder. The rebuild is resumable
before the snapshot is frozen. It saves hourly source provenance, daily coverage,
the new observed source, and `rebuild.json`. Missing outcomes are never imputed;
incomplete matched origins block snapshot creation with a coverage report.
If an upstream gap is later repaired, remove only that day's raw JSON and its
metadata from the new rebuild cache before resuming, so it is fetched again.

### Commands to run from the repository root

Run each block only after the preceding block succeeds. The subshell stops on
the first error. No commands below have been run as part of implementing the fix.

1. Rebuild observations and train the original local benchmark baselines:

```bash
(
set -e
export BENCHMARK_SNAPSHOT=reports/benchmarks/t0_2026_05_26_to_08_23_v3
.venv/bin/python -m src.benchmarks.rebuild_energy
.venv/bin/python -m src.benchmarks.baselines
)
```

The first command makes public ODRE GET requests, not t0 calls. It replaces
neither production CSVs/database rows nor production model artifacts. Once the
snapshot exists, skip the rebuild command. Baseline training refuses an existing
baselines directory; if interrupted, preserve/rename that partial directory
before restarting baseline training. Do not copy v2 models into v3.

2. Verify API covariate sensitivity on corrected inputs and freeze experiments:

```bash
(
set -e
export BENCHMARK_SNAPSHOT=reports/benchmarks/t0_2026_05_26_to_08_23_v3
.venv/bin/python -m src.benchmarks.verify_covariates
.venv/bin/python -m src.benchmarks.validate
.venv/bin/python -m src.benchmarks.run prepare
.venv/bin/python -m src.benchmarks.run local
.venv/bin/python -m src.benchmarks.probabilistic
)
```

The diagnostic sends seven t0 API requests on first execution. Configure
`TFC_API_KEY` in `.env`. Experiment preparation recomputes MASE denominators,
weather statistics and E6 regime thresholds from the corrected pre-period.
Probabilistic training rebuilds quantile models and held-out residual calibration.
All fitting/calibration labels remain strictly before May 26. If interrupted,
skip the already-completed `prepare` step; local forecasts and probabilistic
artifacts resume their existing caches.

3. Run t0 and save/publish the corrected metrics:

```bash
(
set -e
export BENCHMARK_SNAPSHOT=reports/benchmarks/t0_2026_05_26_to_08_23_v3
.venv/bin/python -m src.benchmarks.run remote --live
.venv/bin/python -m src.benchmarks.run report
npm --prefix frontend run build
)
```

This is a fresh 990-request t0 run. Resume with the same remote command after
examining any error; there are no automatic retries of paid requests. Do not
reuse v2 forecasts: corrected historical context and newer consolidated outcomes
can affect comparisons. The hosted model alias may also have changed between
runs. This remains a retrospective benchmark, not a publication-vintage backtest.

Saved metrics: `v3/experiments_v1/metrics.csv`; forecasts:
`v3/experiments_v1/predictions.csv.gz`; dashboard:
`frontend/public/data/benchmark.json` (where `v3` denotes the full directory above).
The report includes original and probabilistic local variants, pinball loss,
interval width and 80% coverage across the same E1–E8 and 7/28/90-day slices.

Optional regression checks for the user to run:

```bash
.venv/bin/python -m pytest tests/test_odre_aggregation.py tests/test_benchmark_validity.py tests/test_benchmark_experiments.py tests/test_t0_benchmark.py -q
```

Production ingestion also uses the corrected aggregator. Previously stored
production CSVs, Supabase rows and models still need their own backfill/export
and retraining; this isolated benchmark rebuild deliberately does not publish to
the operational pipeline. See `docs/supabase_load.md` for historical DB backfill.

## Archived v2 status — superseded by the aggregation audit

**Do not use v2 for new model comparisons.** The original v1 snapshot and successful API
pilot remain intact as historical evidence. The earlier audit below describes v1.

### Observed-data recovery

`src/benchmarks/recover.py` reconstructs missing cells from canonical observations
in an isolated source file. It retains existing nonmissing values. Local national
mix data recovers 170 missing hours in the required input period; the ODRE
consolidated endpoint supplies the remaining two hours on April 30. Twelve-region
weather observations recover all missing weather inputs. These are observed data,
not interpolation. The hourly modeling table had lost rows through its price/mix
join; the energy-only benchmark no longer depends on price availability.

All ten targets and three weather columns now have **zero missing cells** across
the required 180-day input/evaluation span. All **90 origins** are eligible for
**7-, 30- and 90-day contexts**, including historical-weather experiments. No
forward fill is needed in this span. Repairs to earlier training history are also
logged, with input/output hashes and a cell-level recovery ledger.

### Clean baseline provenance

Forty new isolated artifacts were trained: **LightGBM and Ridge × ten targets ×
two modes (calendar / calendar plus historical weather)**. These are explicit
benchmark variants, not the original saved production models. All training labels
end at **2026-05-25 23:00 UTC**, with a strict exclusive May 26 cutoff. Both
algorithms are retained; no winner is selected using benchmark outcomes.

Parameters were fixed in advance (LightGBM follows the repository's existing
600-tree configuration, limited to two CPU threads; Ridge uses alpha 1 with
training-only standardization). No hyperparameter tuning was needed. Features
are built once per daily origin from seven days of past data: prior-day and
prior-week profiles, historical summaries, lead time and calendar variables.
Weather variants add past-weather summaries and prior-day weather profiles.
Every one of the 24 leads uses the same origin cutoff; no within-day actuals enter
later leads. Missing training labels are discarded, never filled. Manifests link
training tables, source data, code and model hashes. A 960-row inference smoke
test passed. Baseline context is fixed at seven days; E4 is a foundation-model
context ablation and must not imply these baselines were trained for each context.

### API covariate behavior

Seven controlled calls used the **same May 25 pre-evaluation origin**, demand
history, 30-day context and 24-hour horizon. An identical repeat call produced
exactly the same medians. Calendar, historical-weather and related-generation
inputs changed predictions; shuffling historical weather or related generation
also changed predictions. This establishes input sensitivity for the tested
configuration, not accuracy improvement. All raw responses and request hashes are
saved. Related-series inputs use documented `hist_variables`; no claim of native
joint attention is made. The adapter now supports `related_history` explicitly.

### Active scope and remaining disclosures

- E2/E5 use historical weather only; no actual future weather is sent.
- Calendar inputs are known across the future horizon.
- Fixed trained baseline ablations requiring other checkpoints are N/A.
- The API alias is not an immutable version guarantee; retain saved responses.
- Foundation-model pretraining overlap remains unknown.
- Historical publication vintages remain unverified; describe results as a
  retrospective zero-shot comparison under stated availability assumptions.
- E8 still needs its fixed scheduler/price configuration during Day 2 setup.

### Reproduction and verification

```bash
# These first two commands deliberately refuse to overwrite existing outputs.
.venv/bin/python -m src.benchmarks.recover
.venv/bin/python -m src.benchmarks.baselines
# Seven bounded API calls on first execution; cached responses on subsequent runs.
.venv/bin/python -m src.benchmarks.verify_covariates
# Read-only evidence verification followed by writing readiness.json:
.venv/bin/python -m src.benchmarks.validate
.venv/bin/python -m pytest tests/test_t0_benchmark.py tests/test_benchmark_validity.py -q
```

Artifacts:
- `reports/benchmarks/t0_recovery_v2/recovery.json`: source recovery evidence.
- `reports/benchmarks/t0_2026_05_26_to_08_23_v2/config.json`: active frozen protocol.
- `reports/benchmarks/t0_2026_05_26_to_08_23_v2/baselines/`: models and manifests.
- `reports/benchmarks/t0_2026_05_26_to_08_23_v2/covariate_diagnostic/summary.json`.
- `reports/benchmarks/t0_2026_05_26_to_08_23_v2/readiness.json`: verified Day 1 status.

Production source CSVs and model artifacts were hash-checked as unchanged.

---

## Original v1 scope and execution

Standalone API inference benchmark. No fine-tuning, promotion, scheduled refresh,
production-model replacement, or dashboard modifications in Day 1.

```bash
# From the repository root; existing dependencies suffice.
.venv/bin/python -m src.benchmarks.t0 prepare
.venv/bin/python -m src.benchmarks.t0 pilot
# Configure TFC_API_KEY in local .env, then send at most four requests:
.venv/bin/python -m src.benchmarks.t0 pilot --live
.venv/bin/python -m pytest tests/test_t0_benchmark.py -q
```

Preparation refuses to overwrite an existing snapshot. Use a different benchmark
ID/output path for a revised protocol. Outputs live under
`reports/benchmarks/t0_2026_05_26_to_08_23_v1/` (ignored by git).
The pilot verifies file hashes before running. Successful responses are cached
by model, endpoint and request payload; retries reuse completed requests.
There are no automatic retries of paid POST requests. Invalid responses and HTTP
errors stop the pilot. Request bodies contain public energy history, no API key.
Keep the key out of git; `.env` is already ignored.

## Frozen protocol

- Evaluation: May 26–August 23, 2026 inclusive, 90 complete UTC days.
- Origins: 00:00 UTC daily; first forecast timestamp equals the origin.
- Targets: consumption, total generation, nuclear, gas, coal, oil, wind, solar,
  hydro and bioenergy, in the source dataset's MWh units.
- Horizon: 24 hours; E3 reports prefixes at 6, 12 and 24 hours.
- Context: 7, 30 and 90 days. Reference context: 30 days.
- Point forecast: median; preserve API mean separately.
- Quantiles: 0.1, 0.5, 0.9.
- Calendar variables: deterministic Europe/Paris hour sine/cosine and weekday.
- Reference weather policy: historical observations only. Future observed weather
  is not passed to the API. Future forecast vintages have not been established.
- Past-input gaps: forward-fill within each context, at most 24 hours, with no
  backfill. Preserve missing values in the frozen dataset and audit counts.
- Outcomes: never imputed. Eligibility requires finite target context after the
  declared fill and finite outcomes for all targets. Use matched origins within
  comparisons and display coverage; do not pool mismatched samples.
- API model: `t0-alpha`. The hosted alias does not expose a guaranteed immutable
  checkpoint. Save responses and run dates; reproducibility is artifact-based.
- Benchmarks represent retrospective availability assumptions, not a claim that
  source publication vintages or pretraining contamination have been verified.

## Experiment specification for Day 2

| Experiment | Fixed comparison | Metrics |
| --- | --- | --- |
| E1 | Frozen t0 vs eligible fixed trained baselines and daily/weekly seasonal naive | Per-target MAE, RMSE, seasonal MASE; training-history denominator, undefined if zero |
| E2 | Target only; calendar; calendar + historical weather; related-series historical covariates after API verification | Paired MAE/MASE differences on common origins |
| E3 | First 6/12/24 leads of identical 24h predictions | Per-lead and cumulative error; not separate horizon-conditioned requests |
| E4 | 7/30/90d contexts with identical inputs otherwise | Accuracy, elapsed request time, matched-origin count |
| E5 | Historical weather: 10%/30% dropout, final 6h outage, noise at 0.1/0.3 pre-period standard deviations; seed 42 | Error changes versus clean historical weather; native missing encoding must first be verified, otherwise label causal imputation explicitly |
| E6 | Demand/temperature above pre-period 95th or below 5th percentile; wind below 5th; freeze thresholds before evaluation | Regime error, paired differences, event/day counts; insufficient evidence labels |
| E7 | Native supported quantiles | Pinball loss, central 80% coverage and interval width; no post-test recalibration |
| E8 | Identical offline carbon accounting, price signal, constraints and ranking weights | Top-5 quality, carbon/cost regret, savings vs immediate execution |

E8 must freeze its price-input provenance and exact scheduler configuration before
execution. It does not use production promotion logic. Fixed feature-based model
ablations that require retraining are N/A. Do not sum marginal source quantiles
and call the result a calibrated aggregate carbon interval.

## Audit findings (September 16, 2026)

- Local modeling data ends August 24 at 12:00 UTC; the partial day is excluded.
- The 180-day input/evaluation grid contains 172 missing hourly rows.
- Each selected weather column has 1,468 missing cells in that grid.
- Eligible origins after the declared input handling: 7d = 90; 30d = 78;
  90d = 18. E4 currently has only 18 matched origins. A full 90-origin context
  comparison requires data repair and a new snapshot; do not silently expand
  interpolation to manufacture full coverage.
- Weather-rich experiments have further coverage limits recorded in origins.csv.
- Saved models vary by target (ridge, LightGBM, random forest, histogram gradient
  boosting and XGBoost). There is no saved LightGBM demand checkpoint.
- Recent supply/demand, source and price reports give a May 11 training cutoff;
  separate consumption/production reports are older (March 31 cutoff). Reports
  cannot prove which training run produced a particular joblib file.
- Model hashes and reported windows are recorded in audit.json, but existing
  files lack linked training manifests. Existing winner selection uses metric
  summaries including test windows. Therefore current checkpoint comparisons
  cannot be labelled untouched-holdout results. Before Day 2 baseline inference,
  either establish checkpoint provenance or train explicitly chosen algorithms
  in a separate benchmark directory using pre-May-26 data and pre-period tuning.
- Archived operational row predictions must not be assumed equivalent to a
  single 24h-origin forecast: within-day lag features can contain newer actuals.
- No TFC_API_KEY was configured at the initial audit; live pilot is pending.

## Pilot and API limitations

The pilot covers three distinct dates, 7/30/90d contexts, a ten-target batch,
calendar covariates and (where eligible) a fourth historical-weather request.
A batch of ten series is not evidence of joint multivariate attention. Related
series can be provided as explicit historical covariates in a later verified
mode; do not claim native joint forecasting from batching alone.

The published API schema includes t0-alpha and covariate fields, but some prose
still lists only older models as supporting covariates. Live acceptance must be
checked; even acceptance alone does not establish that a field influences output.
A paired covariate-ablation pilot is needed before interpreting E2 scientifically.
No null encoding for missing numeric inputs is documented in the inspected
schema; Day 1 rejects residual missing inputs rather than sending guessed values.

API execution keeps foundation-model memory/compute off the Mac. Pilot timing is
end-to-end API latency, not server inference-only time. Server memory cannot be
measured from the client. A full-run estimate remains pending live responses.

Official references inspected September 16, 2026:
- https://docs.retrocast.com/documentation
- https://docs.retrocast.com/documentation/endpoints
- https://docs.retrocast.com/documentation/covariates
- https://api.retrocast.com/openapi

## Day 2 — experiment runner and dashboard

The archived run is `reports/benchmarks/t0_2026_05_26_to_08_23_v2/experiments_v1`.
It uses the now-invalidated v2 snapshot and the isolated baseline artifacts. The original
production forecasting and promotion paths do not call this runner.

```bash
# One-time freeze; refuses to overwrite an existing experiment run.
.venv/bin/python -m src.benchmarks.run prepare
# Cached local baseline forecasts, resumable by configuration.
.venv/bin/python -m src.benchmarks.run local
# Explicit API execution: 990 total requests, less any completed cache entries.
.venv/bin/python -m src.benchmarks.run remote --live
# Optional bounded batch; --limit counts NEW requests, not cached ones.
.venv/bin/python -m src.benchmarks.run remote --live --limit 10
# Recompute metrics / publish local static JSON without making API calls.
.venv/bin/python -m src.benchmarks.run report
npm --prefix frontend run build
```

The remote run is sequential. An HTTP error, timeout, malformed output or quota
error stops it and writes a status record. There are no automatic paid-request
retries. Rerunning resumes completed cases; a request that timed out may have
been processed remotely, so inspect the error before retrying. Each successful
response is saved with receipt timestamp and request hash. Compressed request
bodies are retained and verified during report generation. API secrets are never
written into benchmark artifacts.

### Fixed experiment grid

Eleven t0 configurations × ninety daily origins = **990 requests**. Each request
forecasts all ten targets for 24 hours:

1. Calendar with 30-day context (reference; reused for E1/E3/E6/E7/E8).
2. Target history only with 30-day context.
3. Calendar plus historical weather with 30-day context.
4. Calendar plus related generation/demand histories, as explicit historical
   covariates excluding the target itself, with 30-day context.
5. Calendar with 7-day context.
6. Calendar with 90-day context.
7–11. Historical-weather reference with 10% dropout, 30% dropout, last-six-hour
   outage, noise at 0.1 standard deviations and noise at 0.3 standard deviations.

Weather perturbation uses a deterministic seed per origin. All historical weather
channels are affected; target observations and future calendar inputs remain
unchanged. Dropped values are forward-filled for at most 24 hours, then filled
from a pre-period mean. This measures robustness of the **model plus explicit
imputation policy**, not native support for null inputs. Noise scales and means
are frozen from pre-May-26 data. The same perturbed histories feed local baseline
weather features (whose historical feature context remains seven days).

LightGBM and Ridge run calendar, clean-weather and all five weather-stress
configurations. Daily and weekly seasonal-naive models provide E1/E3/E6/E8
references. Unsupported configurations are explicitly N/A; no extra model training
occurs in Day 2.

### Metrics and comparison rules

- All forecast errors use raw point predictions; t0 point output is its median.
- MAE/RMSE are per target. MASE divides by pre-period weekly seasonal-naive MAE;
  a zero denominator is undefined, not zero error.
- E3 scores cumulative lead ranges 1–6, 1–12 and 1–24 from the same 24h forecast.
  It does not test whether changing the API's requested horizon changes output.
- E6 selects unusual **hours** using pre-period 5th/95th percentiles for demand,
  total production, temperature and wind. The table shows contributing days;
  fewer than ten is labelled limited evidence. This is a regime slice, not a
  claim of a statistically established distribution change.
- E7 reports mean pinball loss across 0.1/0.5/0.9, central 80% coverage, interval
  width and empirical quantile CDFs. It does not label three-quantile pinball loss
  as exact CRPS. Horizon and regime filters are supported.
- Comparisons use the intersection of available model/configuration origins;
  pending configurations remain visible in coverage. Final complete results use
  all ninety origins. Final-7/28-day filters are trailing slices of the fixed
  evaluation period, not rolling live evaluations.
- API latency is per ten-series request including network/serving overhead;
  local latency is per-target predict time excluding loading/feature creation.
  These are labelled separately. Remote peak memory is not exposed by the API.

### Frozen E8 policy

Reuse the project's source-level carbon accounting and ordinal ranking helpers:
**direct operational emissions**, nonnegative source generation for accounting,
**80% carbon / 20% price weights**, one-hour **1 MWh** workload, all 24 hourly
start candidates, and timestamp tie-breaking. Do not apply production uncertainty
penalties, ranking-model overlays, confidence calibration or promotion logic.
Forecast-quality errors still use unclipped generation forecasts.

Every model receives the same **previous-day same-hour price forecast**. Price
actuals only evaluate outcomes. Two missing August 23 price hours were recovered
from Energy-Charts and frozen separately; no production price CSV was changed.
The run protocol stores the price source/recovery hashes and emission factors.
This price reference deliberately avoids assuming archived day-ahead publication
vintages. Energy observations still carry the retrospective availability assumption.

E8 reports combined regret, carbon regret (gCO2e/kWh), cost regret (EUR/MWh),
top-five overlap and capture of the best observed combined-score hour, plus
carbon/cost savings relative to running at the origin. Negative savings are
retained. For the fixed 1 MWh workload, gCO2e/kWh is numerically kgCO2e per job.
No joint carbon prediction interval is derived from marginal source quantiles.

### Dashboard integration and persistence

`frontend/src/BenchmarkView.jsx` reads **only** `/data/benchmark.json`. The new
Benchmarks tab is independent of live recommendation status and requires no API
key or model runtime in the browser. It offers E1–E8 selectors, model/target and
7/28/90-day filters, horizon/regime filters where relevant, error/calibration
charts, outcome tables, scheduling examples, coverage and downloadable protocol.
The public JSON is a distributable frozen artifact intended to accompany the
frontend source in version control. Existing dashboard refresh commands do not
regenerate or remove it. Missing and partial snapshots have explicit UI states.
No scheduled automation or remote deployment is added by Day 2.

Generated detailed artifacts (ignored by git) include compressed predictions,
metric tables, per-day decisions, protocol and raw API responses. Only the compact
public result JSON is needed to render the tab after deployment. A final production
build must follow final report generation so it includes the completed snapshot.

## Probabilistic local variants (implementation added; run manually)

Run these commands in order, stopping on an error. No t0 calls are needed:

```bash
.venv/bin/python -m src.benchmarks.probabilistic
.venv/bin/python -m src.benchmarks.run report
npm --prefix frontend run build
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

The first command fits and forecasts local probabilistic models. It saves resumable
artifacts under `experiments_v1/probabilistic_v1/`; originals and t0 caches stay
intact. The second regenerates `metrics.csv`, `predictions.csv.gz`, per-day decision
results and the public dashboard JSON from all cached predictions.

Methods are fixed before execution:

- **LightGBM · quantile:** three separate regressors with quantile objectives at
  0.1, 0.5 and 0.9, trained on all labels before May 26. Apply row-wise quantile
  sorting to prevent crossing. Other hyperparameters match the fixed benchmark.
- **Ridge · residual quantiles:** refit Ridge and its scaler using only labels
  before April 28, 2026. Reserve April 28–May 25 for held-out calibration, estimate
  residual quantiles separately for each of the 24 forecast leads, and add these
  offsets at inference. Do not refit Ridge on the calibration data afterwards.
- **Daily/weekly naive · residual quantiles:** estimate the same lead-specific
  empirical residual distributions from the pre-evaluation calibration period;
  no regression fit is required. These are probabilistic references.

Residual calibration requires at least 20 complete calibration origins at each
lead. It is an empirical method, not a guarantee of nominal coverage for a time
series. Fit/calibration timestamps, source hashes, parameters and residual offsets
are saved in per-artifact manifests. No evaluation labels enter fitting or
calibration. The held-out calibration period is never adjusted based on the
reported test results.

The Ridge and LightGBM variants cover calendar, historical weather and all five
weather stress configurations. Naive probabilistic references cover the calendar
reference. For all new variants, **q0.5 is the point prediction**, consistently
with t0. This means their MAE/RMSE and scheduling decisions may change; the old
point-only models remain separately labelled rather than silently replacing their
previous scores. No predictions are clipped for forecast/quantile metrics.

The report computes pinball loss (mean over three quantiles), 80% coverage
(q0.1 ≤ actual ≤ q0.9), mean interval width (q0.9 − q0.1), and empirical quantile
coverage. E7 now includes all available probabilistic variants on common origins,
with 6/12/24-hour, 7/28/90-day and regime slices. Its calibration plot displays one
line per probabilistic model. E1/E2/E3/E5/E6 show their probabilistic metrics too.
E4 still concerns t0 context length, and E8 still uses point scheduling rather
than inventing joint carbon intervals from marginal quantiles.

An interrupted probabilistic run resumes completed artifacts/configurations. Report
generation fails clearly if that run exists but has not completed, avoiding a
misleading mixture of partial probabilistic results. Original point-model CSVs
are not modified. Rerunning the probabilistic command skips verified completed
files; edits to the probabilistic implementation require a new output version.
