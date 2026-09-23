# Causal Energy Intelligence Methodology Handbook

## Forecasting Recommendations Causal Inference and Production Operations

Version 1.1  
Current through 20 September 2026

This handbook explains the complete reasoning behind the Causal Energy Intelligence platform. It is written as an interview study guide and as the running record of the methodology. The goal is to make the system understandable after a long break from the project, while preserving enough technical detail to defend each design choice.

The central idea is simple. The platform does not try to win a forecasting competition for its own sake. It tries to answer a decision question: given a flexible electricity workload, which feasible future start times are likely to produce the best carbon and cost outcome, and how certain are we?

The current system has three analytical layers. Forecasting estimates future grid conditions. Recommendation logic compares feasible hours and ranks them. The causal layer asks a harder question: what emissions change would the workload itself cause if moved from one hour to another? These layers are related, but they are not interchangeable.

<!-- pagebreak -->

## How To Use This Handbook

Read Chapters 1 through 4 before an interview if you need the product story and architecture. Read Chapters 5 through 9 for forecasting, ranking, uncertainty, and evaluation. Read Chapters 10 through 13 for carbon accounting and causal inference. Chapters 14 through 17 cover production operations, limitations, interview questions, and the repository map.

When a generated metric conflicts with a number written in an older document, use the latest report artifact as the source of truth. Model performance changes as data, validation windows, and champions change. This handbook focuses on definitions and decision rules that should remain stable.

### Short Interview Summary

The platform schedules flexible electricity workloads into cleaner future hours. It ingests French electricity, weather, and price data; builds leakage-safe hourly features; forecasts consumption, production by source, carbon intensity, and supporting price signals; then ranks feasible workload windows. Models are selected by recommendation regret and ranking quality rather than forecast error alone. Recommendations include uncertainty guards, empirically calibrated confidence, scenario-specific rankings, and historical outcome audits. The causal work currently uses an explicitly labeled marginal-emissions proxy and now has a versioned causal estimand and DAG. The next causal step is an identified marginal-response estimator validated against a European dispatch model.

## Chapter 1 The Product Decision

### The Business Problem

Data centres, EV fleets, batteries, and some industrial processes can choose when to consume electricity. Their work still has to be completed, but the start time may be flexible. The electricity grid changes from hour to hour. The generation mix, demand, price, imports, outages, and renewable availability all change. Moving a workload can therefore change its expected cost and emissions.

The product turns that flexibility into a ranked list of future start times. It does not merely display a forecast chart. It makes a constrained decision recommendation.

### A Restaurant Analogy

Imagine a restaurant that expects twenty customers over an evening. A demand forecast predicts how many customers arrive each hour. A recommendation system goes further: it tells a flexible customer when to arrive to minimize waiting time and price. A causal analysis asks an even harder question: if that customer changes arrival time, which cook, oven, or extra shift actually responds, and how much additional energy does that response consume?

Electricity works similarly. Forecasting describes the likely grid. Ranking chooses an hour. Causal inference estimates the consequence of the workload entering the grid at that hour.

### The Decision Unit

The normalized decision unit is a one MWh workload lasting one hour. This makes results comparable. A real user can later provide workload energy, duration, earliest start, deadline, and maximum delay. Candidate windows must be contiguous and feasible.

The primary target users are data-centre batch workloads and EV or battery charging. These users differ operationally, so the normalized unit is a reference rather than a claim that all workloads behave identically.

## Chapter 2 The End To End Architecture

The operational path is:

```text
Source APIs
  -> canonical hourly data
  -> quality and freshness checks
  -> forecasting features
  -> consumption and production forecasts
  -> source generation forecasts
  -> carbon and price signals
  -> feasible workload windows
  -> scenario and uncertainty aware ranking
  -> top five recommendations
  -> dashboard and history logging
  -> actual outcome audit after hours settle
```

This sequence matters. Each stage produces a contract consumed by the next stage. A recommendation can be wrong because the input data was stale, because an upstream forecast missed, because the ranking objective was poorly specified, or because the estimated carbon basis did not represent the true grid response. Monitoring therefore has to inspect the whole chain.

### Separation Of Responsibilities

| Layer | Main question | Typical output |
| --- | --- | --- |
| Data | What happened and what information is available now | Canonical hourly observations and future exogenous data |
| Forecasting | What grid conditions are likely in future hours | Predicted consumption production price and carbon signals |
| Recommendation | Which feasible hour has the best expected decision outcome | Ranked candidate windows and top five list |
| Causal inference | What emissions change would the workload cause | Treatment effect with uncertainty and evidence tier |
| Monitoring | Is the system fresh accurate and safe to publish | Health drift audit and promotion reports |

The recommendation layer may use forecasts without making a causal claim. The causal layer requires an intervention definition, identification assumptions, and counterfactual reasoning.

## Chapter 3 Data Sources And Contracts

### Electricity Data

Historical and near-real-time French electricity consumption and production come from ODRE Opendatasoft eco2mix datasets. The source data is available nationally and regionally at 15-minute resolution. The current scheduled cloud path stores national France data by default to control Supabase storage usage.

The source values are power in MW. Four 15-minute observations are converted into hourly energy:

```text
hourly MWh = sum of MW observations multiplied by 0.25 hours
```

The production mix includes nuclear, gas, coal, oil, wind, solar, hydro, and bioenergy signals used by forecasting and carbon accounting.

### Weather Data

Historical and future weather come from Open-Meteo. Regional observations use representative French cities and are aggregated into compact national features for the current forecasting path. Weather matters because temperature affects demand, while wind speed and solar conditions affect renewable production.

### Price Data

French day-ahead prices come primarily from Energy-Charts for bidding zone FR. ENTSO-E remains a fallback. Price is a supporting signal. The product does not claim that exact spot-price prediction is its main objective.

### Storage And Canonical Schema

The ingestion layer converts source-specific payloads into stable internal contracts. This protects downstream code from API response changes. Supabase stores transformed, model-relevant rows. Raw payloads are not retained by default. Local processed CSV files support development and model training.

### Time Discipline

All analytical timestamps are normalized to UTC. The dashboard may present user-friendly times, but model joins, candidate windows, history, and audits use timezone-aware UTC timestamps. This prevents daylight-saving changes from creating duplicate or missing local hours.

### Data Quality Checks

The pipeline checks file existence, row counts, latest timestamps, duplicate timestamps, missing hours, required columns, nulls, negative production or consumption, dashboard presence, and recommendation count. Stale source data is a critical failure in normal production mode.

Analogy: the model is a kitchen, but data contracts are the labelled ingredients. Even an excellent chef cannot recover if salt arrives in a sugar container or yesterday's food is presented as fresh.

## Chapter 4 Forecast Time Integrity

### The Leakage Problem

A forecast must use only information available when the forecast is made. Using realized generation from the future would produce impressive offline accuracy and fail in production. This is called target leakage or look-ahead bias.

The strict forecasting dataset therefore uses calendar information, lagged values, rolling summaries, and upstream forecasts. Contemporaneous realized electricity and weather columns are preserved for causal analysis and auditing, but strict forecasters do not use them unless they are replaced by values genuinely available at prediction time.

### Main Feature Families

- Calendar features such as hour, day of week, month, weekend, peak period, morning ramp, evening peak, and cyclic encodings.
- Price lags from one hour through two weeks, rolling means, rolling dispersion, and momentum features.
- Consumption and production lags and rolling signals.
- Source-specific generation lags for nuclear, fossil generation, wind, solar, hydro, and bioenergy.
- Forecasted future consumption, total production, residual demand, and supply-demand gap.
- Forecast weather for the operational future horizon.

### Why Cyclic Encoding Helps

Hour 23 and hour 0 are numerically far apart but are adjacent on a clock. Sine and cosine encodings place hours around a circle, allowing models to learn smooth daily and yearly cycles.

### Upstream Forecasts

The system forecasts consumption, total production, and source-level production before building carbon and recommendation signals. This is similar to planning a road trip. You first estimate traffic and fuel availability, then choose a route. A route recommendation built directly from unknown future traffic would be incomplete.

## Chapter 5 Forecasting Models And Validation

### Candidate Models

The model set includes a 24-hour naive baseline, ridge regression, random forest, histogram gradient boosting, LightGBM, and XGBoost. The naive model is important because a complex model must beat a simple rule to justify its cost.

### Walk Forward Validation

Random train-test splits are inappropriate for time series because they mix past and future. The platform uses expanding walk-forward windows. Each validation period is predicted using only earlier data. The latest test period is dynamically aligned to the newest data.

Analogy: to test whether someone can forecast next week's weather, we repeatedly place them at historical dates and hide everything that happened afterward. Giving them random pages from the future would invalidate the test.

### Central Forecast Metrics

MAE measures the average absolute forecast error in the target unit. RMSE penalizes large misses more heavily. sMAPE expresses error relative to magnitude and remains useful across differently scaled targets. Directional accuracy checks whether the price movement direction was predicted correctly.

These metrics diagnose forecast quality, but they do not fully measure recommendation quality. A model may miss every value by a constant amount and still rank the hours correctly. Another model may have low MAE while swapping the best and second-best hours.

### Quantile Forecasts

The production forecasting contract is probabilistic rather than point-only. For price, consumption, total production, and each generation source, the model emits three values: `q10`, `q50`, and `q90`. The median `q50` is the central forecast used for ranking. The range from `q10` to `q90` is a nominal 80 percent predictive interval.

An analogy is a travel-time estimate. Saying a trip will take 40 minutes is a point forecast. Saying there is a central estimate of 40 minutes, with an 80 percent interval from 32 to 55 minutes, is more useful for deciding whether a tight appointment is safe.

The current implementation calibrates quantiles from residuals produced by earlier walk-forward predictions. At each historical prediction time, only errors already observed are available for calibration. Persisted production models use out-of-sample walk-forward residuals. This avoids using the outcome of the row being evaluated to construct its own interval.

Quantile forecasts do not automatically guarantee uncertainty quality. The system therefore reports pinball loss for each quantile, mean pinball loss, observed 80 percent coverage, coverage error relative to 80 percent, mean and median interval width, normalized interval width, below-interval and above-interval rates, and the Winkler interval score. Pinball loss rewards accurate quantile placement. Coverage detects intervals that are too narrow. Width detects intervals that are technically safe but unhelpfully broad. The Winkler score penalizes both broad intervals and observations that fall outside them.

### Ranking Metrics

Top-1 hit rate asks whether the first recommendation was truly best. Top-3 or top-5 capture asks whether the actual best hour appeared in the recommended shortlist. Pairwise ranking loss measures how often the model orders two candidates incorrectly. Spearman correlation measures overall rank agreement.

These metrics align with the product question. The user needs a good hour, not merely a numerically accurate chart.

## Chapter 6 Carbon Accounting

### Average Carbon Intensity

For each source, generation is multiplied by its configured emissions factor. Total emissions are then divided by total generation:

```text
total emissions kg CO2e
  = sum of source generation MWh multiplied by source factor kg CO2e per MWh

carbon intensity g CO2e per kWh
  = total emissions kg CO2e divided by total generation MWh
```

The units are numerically equivalent because one kg per MWh equals one g per kWh.

### Direct And Lifecycle Boundaries

Direct operational emissions represent emissions produced during electricity generation. Lifecycle factors additionally include upstream construction, fuel extraction, transport, and related processes. The primary product and causal estimand use direct operational emissions. Lifecycle accounting may be reported separately but must not be mixed silently into the same result.

### Why Configuration Matters

Emission factors live in configuration rather than calculation code. This makes the accounting boundary explicit and reviewable. It also prevents a hidden code change from silently redefining carbon performance.

## Chapter 7 Recommendation Ranking

### Candidate Window Construction

The system converts hourly signals into feasible workload windows. For a three-hour workload, a candidate is valid only if three consecutive hourly rows exist. Earliest start, latest end, maximum delay, and duration constraints remove infeasible options before ranking.

### Rank Normalization

Price and carbon have different units and scales. The platform ranks each candidate within its decision group, then normalizes the ranks to the interval from zero to one. The best rank becomes zero and the worst approaches one.

The base predicted score is:

```text
base score
  = price weight multiplied by normalized predicted price rank
  + carbon weight multiplied by normalized predicted carbon rank
```

Lower is better. Rank normalization prevents a large numerical price scale from overwhelming carbon merely because EUR per MWh and g CO2e per kWh use different units.

### Scenario Presets

| Scenario | Price weight | Carbon weight | Business interpretation |
| --- | ---: | ---: | --- |
| Emissions reduction | 0.2 | 0.8 | Prioritize cleaner hours while retaining a limited cost signal |
| Balanced operations | 0.5 | 0.5 | Treat cost and carbon rank equally |
| Budget control | 0.7 | 0.3 | Prioritize cost while retaining a carbon preference |

The scenario selector reranks the same feasible candidates. It does not retrain the forecasting models.

### Ranking Model Overlay

A ranking-specific classifier can estimate the probability that a candidate belongs in the actual top five. The learned overlay is accepted only when out-of-window combined regret and carbon regret do not deteriorate relative to the transparent baseline score. Day-level diagnostics show where it helped and where it regressed.

This acceptance rule reflects a general principle: added model complexity must improve the decision, not only an intermediate prediction.

## Chapter 8 Uncertainty And Confidence

### Decision Uncertainty

Candidates are uncertain when predicted price or carbon values are too close to neighbouring candidates. Small score margins mean that modest forecast errors could reverse the order. The system converts this separation into an uncertainty score.

Candidates above the uncertainty threshold receive a ranking penalty. If no low-uncertainty candidate exists, the output explicitly says that no low-risk recommendation is available. The system does not hide this condition behind a normal-looking rank.

Analogy: if two runners finish within a camera frame, announcing the winner with the same confidence as a ten-second victory would be misleading.

### Prediction Intervals

Each operational candidate carries its own price and carbon `q10-q90` interval. Source-generation quantiles are propagated through direct-operational emission factors to produce carbon-intensity quantiles. Ranking uses the median, while the interval width contributes to the uncertainty score and can trigger the high-uncertainty guard. A model-wide historical residual width remains only as a compatibility fallback for older artifacts.

An 80 percent interval should contain the settled actual value about 80 percent of the time over a sufficiently large, representative sample. Coverage much below 80 percent indicates overconfidence. Coverage far above 80 percent can indicate intervals that are too wide to support useful decisions. Coverage should also be inspected by lead hour, season, and grid regime because good aggregate calibration can hide weak conditional calibration.

### Heuristic Confidence

Initial confidence combines recommendation rank, score margin, and cross-model agreement. A first-ranked candidate receives a stronger rank component. A candidate well separated from alternatives receives a stronger margin component. An hour selected by several models receives a stronger agreement component.

### Empirical Calibration

Heuristic confidence is then mapped to historical top-five hit rates and observed regret. Minimum sample-size guards prevent tiny bins from being treated as reliable evidence. Scenario-specific calibration is used because a carbon-first ranking and a budget-first ranking may have different historical reliability.

Confidence is therefore a statement about historical recommendation reliability, not a probability that every forecasted value is exactly correct.

## Chapter 9 Recommendation Evaluation

### Regret

Regret measures the cost of the chosen decision relative to the best decision that was actually available after outcomes are known.

```text
combined regret
  = actual score of recommended hour
  - actual score of best feasible hour
```

Carbon regret compares the recommended hour with the actual cleanest feasible hour. Cost regret compares it with the actual cheapest feasible hour. Zero regret means the recommendation matched the realized optimum under that objective.

Analogy: forecast error asks how accurately every menu price was predicted. Regret asks how much worse the chosen meal was than the best meal that could have been ordered. The second question is closer to the user's decision.

### Champion Selection

The champion score is lower-is-better and currently combines:

| Component | Weight |
| --- | ---: |
| Realized recommendation regret | 35 percent |
| Carbon regret | 25 percent |
| Top-five ranking loss | 20 percent |
| Price-direction error | 10 percent |
| Carbon-intensity error | 10 percent |

This prevents forecast MAE from becoming the sole model-selection criterion. Guarded metrics also prevent a candidate from winning the weighted average by materially damaging recommendation or carbon regret.

### Policy Backtesting

Historical policy backtests evaluate the exact rank-one recommendation emitted by each model and scenario. They report top-one and top-five hit rates, realized regret, confidence, and whether the uncertainty guard could find a low-risk option.

### Previous Day Outcome Audit

Operational recommendations are appended to history. Once actual data arrives, the audit joins each past recommendation to observed price, production, consumption, and carbon intensity. It recomputes actual ranks and regret. This distinguishes offline validation from genuine production behavior.

## Chapter 10 Average And Marginal Emissions

### Average Emissions

Average carbon intensity divides all current generation emissions by all current generation. It answers an attribution question: how carbon-intensive was the electricity mix on average?

### Marginal Emissions

Marginal emissions ask which generators, imports, storage assets, or curtailment decisions respond to an incremental change in demand. This is the more relevant concept for a flexible workload because the workload does not consume a proportional slice of every generator.

Analogy: the average cost of feeding a group includes the rent, staff, and every dish already prepared. The marginal cost of one extra guest depends on what the kitchen must cook next. Average and marginal values answer different questions.

### Current Marginal Emissions Proxy

The current MVP examines positive hour-to-hour generation changes by source. It treats the increasing source mix as the responding marginal mix:

```text
marginal proxy intensity
  = sum of positive source generation change multiplied by source emissions factor
  / sum of positive source generation change
```

The proxy records its dominant source, response shares, coverage, and a coarse confidence label. If no usable response is available, recommendation logic falls back explicitly to average carbon rather than hiding the missing evidence.

### What The Proxy Cannot Prove

Generation can change because of demand, weather, outages, exports, storage, market schedules, ramp constraints, and renewable curtailment. Positive generation changes are therefore not automatically caused by a workload. The current causal-adjusted dashboard basis remains a marginal-emissions proxy, not an identified treatment effect.

The proxy is still useful for comparing average-carbon and marginal-carbon rankings. The platform measures top-one changes, top-five overlap, absolute rank displacement, proxy coverage, and regret differences.

## Chapter 11 The Causal Estimand

### The Exact Question

The primary estimand is the conditional short-run emissions effect of moving a flexible workload from a baseline hour to a candidate hour.

```text
effect of shift from a to b
  = expected total system emissions under the shifted schedule
  - expected total system emissions under the baseline schedule
```

Negative values indicate that the shift reduces emissions. The normalized treatment moves a one MWh, one-hour workload while conserving total energy.

### Baseline Definition

The baseline defaults to the exact dashboard access timestamp because the product does not know the user's planned start time. A future dashboard field will allow the user to enter a planned start, which then overrides dashboard access time. The precise timestamp must be stored in UTC.

This resolves the ambiguity of the phrase run now. A comparison is meaningful only when the reference time is explicit.

### Outcome Boundary

The outcome is total direct operational emissions across France and responding interconnected European bidding zones. A French workload may change imports or exports, so French domestic generation alone is not a complete causal boundary.

The primary response horizon runs through workload completion plus six hours. Six, twelve, and twenty-four-hour windows are retained as sensitivity checks for delayed ramping, storage, balancing, and cross-border effects.

### Product Objective

The future causal recommendation objective is to maximize confidently avoided kg CO2e. The decision statistic should use a conservative lower confidence bound of avoided emissions. Price, service-level risk, and feasibility remain constraints or secondary objectives rather than being mixed into the causal effect itself.

## Chapter 12 The Causal Graphs

### Grid Effect Graph

```text
pre-decision forecasts prices outages capacities and storage state
  -> baseline grid state
  -> workload timing

workload timing
  -> net load
  -> generator dispatch imports storage curtailment and balancing
  -> total operational emissions
```

The pre-decision adjustment set includes calendar, weather forecast, baseline-load forecast, renewable-generation forecast, known outages, fuel and carbon prices, interconnector capacity, and initial storage state.

Net load, dispatch, cross-border flows, storage dispatch, curtailment, and balancing are mediators. Conditioning on them while estimating the total effect can block part of the effect we are trying to measure.

Analogy: if exercise affects health partly through weight loss, controlling for weight loss removes part of exercise's total effect. Dispatch plays a similar mediator role between workload timing and emissions.

### Product Effect Graph

The effect of showing a recommendation is a separate question:

```text
recommendation shown
  -> recommendation accepted
  -> realized workload shift
  -> grid response
  -> emissions
```

User constraints influence acceptance and realized timing. Estimating this product effect requires exposure logs, acceptance, actual workload telemetry, and user constraints. The current system defines this graph but does not estimate the product effect.

### Contract Validation

The versioned contract validates that the graph is acyclic, that a treatment-to-outcome path exists, and that adjustment variables occur before treatment. It rejects treatment descendants and mediators in the adjustment set. The current contract is a design artifact. It does not transform the marginal proxy into a causal estimate.

## Chapter 13 Building A Stronger Causal Estimator

### Required Data Expansion

The next causal stage needs actual and forecast load, renewable forecasts and errors, cross-border physical and scheduled flows, outages, balancing activation, storage state, curtailment, interconnector capacity, fuel prices, EU ETS prices, and preferably more granular dispatch. ENTSO-E is the main European source for this expansion.

The first ENTSO-E data contract covers France and its directly connected bidding zones. It stores
directional physical flows, scheduled exchanges, transfer capacity, versioned load and renewable
forecasts, outage revisions, and native-resolution balancing data. Operational forecast snapshots
can be joined to later settled actuals to measure what the system genuinely knew at decision time.
Historical downloads are marked `historical_final`: they help exploratory modeling, but they do not
prove that the same forecast vintage was available at a past decision time.

Analogy: an operational snapshot is a photograph taken before the match; a historical final value is
the edited match report. Both contain useful information, but only the photograph can establish what
was visible before play began.

### Constrained Marginal Response Regression

The first estimator should model how generation technologies and connected zones respond to changes in French net load while controlling for pre-decision conditions. Physical constraints can regularize the estimates. Total marginal response should approximately balance the demand change, subject to storage, losses, and imports. Sign and monotonicity assumptions must be tested by regime rather than imposed blindly.

### Double Machine Learning

Double or debiased machine learning can estimate a low-dimensional treatment effect while flexible models learn the treatment and outcome nuisance functions. Orthogonal scores reduce sensitivity to nuisance-model error. Time-blocked cross-fitting is required because random folds would leak temporal structure.

Double machine learning does not create identification. Its causal interpretation still depends on the adjustment set, overlap, measurement, and absence of important unobserved confounding.

### Structural Dispatch Counterfactual

A European dispatch or unit-commitment model can simulate the grid with and without the workload. PyPSA-Eur is a suitable open foundation. This approach represents transmission, generator constraints, storage, and cross-border response more directly than a simple statistical proxy.

The strongest architecture triangulates an econometric estimate, a flexible DML estimate, and a structural dispatch counterfactual. Agreement increases confidence. Disagreement becomes model-risk evidence and may suppress the recommendation.

### Falsification And Sensitivity

Future workload changes should not explain earlier emissions. Lead and placebo tests should therefore be near zero. Negative controls can reveal residual confounding. Omitted-variable sensitivity analysis should report how strong an unmeasured confounder must be to overturn the result. Effects should also be stable across seasons, congestion states, outages, and renewable regimes.

### Evidence Tiers

| Tier | Meaning | Permitted claim |
| --- | --- | --- |
| Average | Attributional grid mix | Average carbon intensity |
| Proxy | Marginal response heuristic | Proxy-adjusted recommendation |
| Observational causal | Identified estimator with diagnostics | Estimated short-run effect under stated assumptions |
| Structural | Calibrated dispatch counterfactual | Simulated effect under model assumptions |
| Validated | Triangulated and operationally audited | Production causal recommendation with uncertainty |

## Chapter 14 Production Orchestration

### Daily Modular Workflow

The GitHub workflow separates ingestion, monitoring, optional retraining, publishing, and deployment. This avoids repeating a long retraining job when only a later publishing step fails.

```text
ingest
  -> preflight monitoring
  -> retrain only when required
  -> publish recommendations and dashboard
  -> deploy
```

GitHub cron uses UTC and may start late. The workflow proceeds whenever GitHub starts it instead of rejecting the run because a local-time guard was missed.

### Retraining Decision

Preflight monitoring requests retraining when model artifacts are missing or monitoring reports material degradation. Otherwise, the workflow reuses the current champion and refreshes only future data, recommendations, monitoring, and the dashboard.

### Promotion Gate

Before retraining, the system snapshots production artifacts. A candidate trains separately. It is promoted only if its weighted lower-is-better score beats the incumbent and guarded regret metrics remain acceptable. A failed or rejected candidate restores the incumbent snapshot.

Settled operational outcomes are advisory until enough decision days exist. After the minimum evidence threshold, poor top-five hit rate or excessive carbon regret can block promotion.

Analogy: a football team does not replace its starting goalkeeper because a new player performed well in one drill. The candidate must improve the relevant match outcomes and must not fail critical safety checks.

## Chapter 15 Dashboard Interpretation

### Trust And Freshness

The dashboard combines pipeline health, forecast monitoring, operational audit readiness, generation time, and the active reference hour. A recommendation without fresh data should not look equivalent to one built from current inputs.

### Scenario And Basis

Scenario changes the price-carbon preference. Basis changes the carbon concept. The standard basis uses average-carbon rankings. The current causal-adjusted MVP uses the marginal-emissions proxy. Scenario-aware causal artifacts ensure both selectors affect the recommendation list.

### Carbon Saving Reference

Carbon saving must name its comparison timestamp. The current-reference field is preferable to a vague run-now label. Once the planned-start input is connected, that user value will override dashboard access time.

### Previous Day Audit

The audit view should answer four questions. What did the system recommend? What actually happened? Where did the recommended hour rank after outcomes settled? How much regret or saving resulted relative to the baseline and actual optimum?

## Chapter 16 Limitations And Honest Claims

### Current Limitations

- Scheduled cloud ingestion primarily stores national France electricity data, so regional claims are not yet supported by the operational path.
- The marginal-emissions method is a proxy based on generation changes, not an identified treatment effect.
- Cross-border and balancing data are not yet fully represented in the causal estimator.
- Confidence calibration is limited by the number and diversity of settled historical recommendations.
- Forecasts can fail during rare negative-price events, outages, and regime changes.
- A normalized one MWh result may not scale linearly to a large battery or data-centre shift.
- User compliance and actual workload telemetry are not yet available, so the product effect is not identified.

### Claims We Can Defend Today

We can say that the platform produces leakage-aware, uncertainty-guarded, scenario-specific workload recommendations and evaluates them with ranking metrics and realized regret. We can say that it compares average-carbon rankings with a marginal-emissions proxy. We can say that the causal estimand and DAG are explicitly defined and validated as a design contract.

We should not yet say that the dashboard proves the workload caused a specific emissions reduction. That claim requires the identified estimator, cross-border data, uncertainty interval, and validation described in Chapter 13.

## Chapter 17 Interview Questions And Answers

### Why Did You Optimize Ranking Instead Of Only MAE

The product chooses an hour. MAE measures numerical forecast error across all hours, but it does not guarantee that the best hours are ordered correctly. I therefore retained MAE as a diagnostic and selected models using regret, carbon regret, pairwise ranking loss, top-five quality, direction accuracy, and carbon error. This aligns model selection with the user's decision.

### What Is Recommendation Regret

Regret is the realized loss from choosing the recommended hour instead of the best feasible hour after actual outcomes are known. It translates model quality into decision cost. Carbon regret and cost regret show which objective was missed.

### How Did You Prevent Leakage

Strict forecasters use calendar features, lagged and rolling historical values, upstream forecasts, and future exogenous values available at forecast time. Realized contemporaneous electricity variables are preserved for causal analysis and auditing but excluded from strict forecasting. Validation uses expanding time windows rather than random splits.

### Why Use Top Five Recommendations

A single recommendation is brittle and may conflict with user constraints that the system does not know. A top-five list preserves useful alternatives, supports user choice, and allows top-k capture evaluation. The rank-one option remains auditable.

### How Is Confidence Calculated

The initial score combines rank, separation from nearby candidates, and agreement across models. Historical calibration then maps confidence bins to observed top-five hit rates and regret. Minimum row thresholds prevent small samples from producing overconfident scores.

### What Is The Difference Between Average And Marginal Carbon

Average carbon describes the whole generation mix. Marginal carbon describes the resources that respond to an incremental demand change. Workload shifting is a consequential decision, so marginal carbon is conceptually closer to the question. The current implementation is explicitly labeled as a proxy because observed generation changes are not automatically caused by load.

### What Makes A Causal Estimate Credible

A credible estimate starts with a precise intervention, comparator, outcome, population, time horizon, and spatial boundary. It requires a defensible DAG and pre-treatment adjustment set, overlap, uncertainty intervals, placebo tests, sensitivity analysis, and validation against another method or experiment. A sophisticated algorithm cannot replace identification.

### Why Include Interconnected European Zones

An incremental French workload can change imports and exports. Measuring only French domestic generation could move emissions outside the boundary rather than eliminate them. The causal outcome therefore covers France and responding connected zones.

### How Do You Decide Whether To Retrain

The daily process first checks freshness and recent operational performance. Retraining occurs only when monitoring recommends it or required model artifacts are missing. A candidate must pass a promotion gate against the incumbent. This reduces compute and prevents a bad retrain from replacing a working production model.

### How Would You Validate The Product Effect

I would log recommendation exposure, accepted hour, actual workload energy, duration, user constraints, and completion. For near-tied safe hours, randomized encouragement can create stronger evidence about acceptance and realized shifting. Grid emissions attribution would still use the identified grid-effect estimator because a single small workload is difficult to detect directly in national emissions.

## Glossary

| Term | Plain meaning |
| --- | --- |
| Actual | Value observed after the hour has occurred |
| Baseline | Schedule against which a candidate shift is compared |
| Candidate window | Feasible contiguous start and duration option |
| Carbon intensity | Emissions per unit of electricity |
| Champion | Production model selected by the decision-quality rule |
| Confounder | Pre-treatment variable affecting both treatment assignment and outcome |
| Counterfactual | Outcome that would occur under an alternative intervention |
| DAG | Directed acyclic graph representing assumed causal relationships |
| Estimand | Exact causal quantity the analysis intends to estimate |
| Forecast | Prediction of a future observable value |
| Marginal emissions | Emissions caused by a small or defined change in electricity demand |
| Mediator | Variable through which the treatment affects the outcome |
| Overlap | Comparable data exists for the treatment choices being evaluated |
| Proxy | Measurable substitute that approximates a harder quantity |
| Regret | Realized gap between the chosen decision and the best feasible decision |
| Scenario | Business preference that changes price and carbon weights |
| Treatment | Intervention whose causal effect is being estimated |

## Repository Guide

| Topic | Primary location |
| --- | --- |
| Data contracts | `src/data/contracts.py` and `docs/data_contracts.md` |
| Data sources | `docs/data_sources.md` |
| Feature construction | `src/features/price_features.py` |
| Forecast training | `src/models/train_forecast.py` |
| Future recommendations | `src/models/future_recommendations.py` |
| Carbon accounting | `src/carbon/intensity.py` |
| Marginal proxy | `src/carbon/marginal.py` |
| Workload ranking | `src/optimization/workload_shift.py` |
| Causal recommendations | `src/causal/recommendations.py` |
| Estimand contract | `config/causal_estimand.json` and `src/causal/estimand.py` |
| Causal DAGs | `src/causal/dag.py` and `docs/causal_dag.md` |
| Forecast monitoring | `src/monitoring/forecast_monitor.py` |
| Outcome audit | `src/monitoring/recommendation_outcome_audit.py` |
| Promotion gate | `scripts/gated_retrain.py` |
| Dashboard contract | `scripts/build_dashboard_data.py` |
| Daily workflow | `.github/workflows/daily-ingestion-monitor.yml` |

## Running Document Update Checklist

Update this handbook when any of the following changes:

1. Data source, geographic resolution, storage policy, or timestamp convention.
2. Forecast target, feature availability rule, validation design, or model family.
3. Recommendation score, scenario weight, uncertainty threshold, confidence calibration, or champion rule.
4. Carbon accounting boundary or emission-factor methodology.
5. Causal treatment, comparator, outcome, adjustment set, estimator, or evidence tier.
6. Operational audit, retraining trigger, promotion gate, or deployment workflow.
7. Dashboard terminology that changes the meaning of a metric.

Regenerate the handbook after editing this Markdown source:

```text
make methodology-handbook
```

## Research References

Athey S Tibshirani J and Wager S. Generalized Random Forests. Annals of Statistics 2019. https://doi.org/10.1214/18-AOS1709

Athey S and Wager S. Policy Learning with Observational Data. Econometrica 2021. https://doi.org/10.3982/ECTA15732

Chernozhukov V and others. Double Debiased Machine Learning for Treatment and Structural Parameters. The Econometrics Journal 2018. https://doi.org/10.1111/ectj.12097

Cinelli C and Hazlett C. Making Sense of Sensitivity Extending Omitted Variable Bias. Journal of the Royal Statistical Society Series B 2020. https://doi.org/10.1111/rssb.12348

Dudik M Langford J and Li L. Doubly Robust Policy Evaluation and Learning. ICML 2011. https://icml.cc/2011/papers/554_icmlpaper.pdf

Hawkes A D. Estimating Marginal CO2 Emissions Rates for National Electricity Systems. Energy Policy 2010. https://doi.org/10.1016/j.enpol.2010.05.053

Holland S P Mansur E T Verdier V and Yates A J. Regularization from Economic Constraints A New Estimator for Marginal Emissions. NBER Working Paper 32065 2024. https://doi.org/10.3386/w32065

Huber J and others. The Effect of Price Based Demand Response on Carbon Emissions in European Electricity Markets. Applied Energy 2021. https://doi.org/10.1016/j.apenergy.2021.117040

Horsch J Hofmann F Schlachtberger D and Brown T. PyPSA Eur An Open Optimisation Model of the European Transmission System. Energy Strategy Reviews 2018. https://doi.org/10.1016/j.esr.2018.08.012

Lipsitch M Tchetgen Tchetgen E and Cohen T. Negative Controls A Tool for Detecting Confounding and Bias in Observational Studies. Epidemiology 2010. https://doi.org/10.1097/EDE.0b013e3181d61eeb

Siler Evans K Azevedo I L Morgan M G and Apt J. Regional Variations in the Health Environmental and Climate Benefits of Wind and Solar Generation. Proceedings of the National Academy of Sciences 2013. https://doi.org/10.1073/pnas.1221978110

Validity Ranges of Locational Marginal Emission Factors for the Environmental Assessment of Electric Load Shifting. Applied Energy 2025. https://doi.org/10.1016/j.apenergy.2024.125069
