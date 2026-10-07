# Causal Estimand and DAGs

The versioned causal contract lives in `config/causal_estimand.json`. Validate it and write the
machine-readable artifact with:

```bash
make causal-contract
```

The output is `reports/causal/estimand_spec.json`. It contains the estimand, the grid-effect DAG,
the separate product-effect DAG, and validation results. The current marginal-emissions proxy does
not become an identified causal estimate merely by satisfying this design contract.

## Primary Estimand

The primary causal question is the conditional short-run change in total operational emissions
caused by moving an energy-conserving workload from a baseline start time to a candidate start time:

```text
tau(a, b, q, d | X) =
  E[emissions under shift(a -> b, q, d) - emissions under baseline(a, q, d) | X]
```

- `a` is the baseline start. It defaults to the exact dashboard access timestamp; an explicit user
  planned start overrides it.
- `b` is a feasible candidate start within the next 24 hours.
- `q` is workload energy. The normalized contract uses 1 MWh.
- `d` is workload duration. The normalized contract uses one hour.
- `X` contains only pre-decision grid information.
- The outcome is measured in `kg_co2e` across France and responding interconnected European bidding
  zones using direct operational emissions.
- The primary product objective is to maximize the lower confidence bound of avoided emissions.

The primary target profiles are data-centre batch workloads and EV/battery charging. Actual
workload size and duration must replace the normalized values before claims are made for a specific
user workload.

## Grid-Effect DAG

```text
calendar -------------------------+
weather forecast -----------------|
baseline load forecast -----------|
renewable generation forecast ----|
known generation outages ---------|--> baseline grid state --> workload timing
fuel price ------------------------|             |                    |
carbon price ----------------------|             |                    v
interconnector capacity -----------|             +-------------> grid response
initial storage state -------------+                                  |
                                                                     v
workload timing --> net load --> dispatch / flows / storage --> total operational emissions
```

The pre-decision adjustment set is:

- Calendar
- Weather forecast
- Baseline-load forecast
- Renewable-generation forecast
- Known generation outages
- Fuel and carbon prices
- Interconnector capacity
- Initial storage state

Net load, generator dispatch, cross-border flows, storage dispatch, renewable curtailment, and
balancing activation are post-treatment mediators. They are excluded from the adjustment set when
estimating the total workload-shift effect.

## Product-Effect DAG

The effect of showing a recommendation is a different causal question:

```text
recommendation shown
  -> recommendation accepted
  -> realized workload shift
  -> grid response
  -> total operational emissions

user constraints -> recommendation accepted
user constraints -> realized workload shift
```

This estimand requires recommendation exposure, acceptance, actual workload telemetry, and user
constraints. Collection is implemented through the dashboard and
`observed_workload_decisions`, but the effect is not estimated until there are enough completed
observations and the overlap, power, telemetry-quality, and sensitivity gates pass.

## Observed Treatment Contract

The unit of analysis is one workload scheduling decision. The dashboard creates a UUID and records:

- Baseline: dashboard access time, optional user-planned start, chosen baseline source, and resolved
  baseline timestamp.
- Recommendation exposure: generation time, basis, scenario, rank-1 start, and the complete set of
  candidate alternatives shown to the user.
- Choice: selected start and whether it was the recommendation, another candidate, or a custom time.
- Execution: actual start, actual completion, and metered or reported energy consumed.
- Workload context: planned energy, duration, type, earliest start, latest completion, and maximum
  delay.

The raw table and decision-level analysis view are created by
`db/observed_workload_treatment.sql`. Direct personal identifiers are deliberately excluded. The
browser calls `frontend/api/workload-decisions.js`; this server-side function is the only dashboard
component that receives the Supabase service-role credential.

Run `make treatment-readiness` to write
`reports/metrics/observed_treatment_readiness.json`. This collection remains separate from the
national hourly feature mart because a workload decision and a grid hour are different units of
analysis. They are joined point-in-time only when an estimator dataset is built.

## First Identified Estimator

`make causal-estimator` implements `constrained_marginal_response_v1` from
`config/causal_estimator.json`. For each completed workload decision, it expands the exact candidate
set, baseline hour, and executed hour into a decision-hour panel. Actual energy in MWh is the dose;
unexecuted feasible alternatives are controls. Alternatives whose response windows overlap the
actual execution are excluded from fitting to avoid labeling exposed hours as untreated.

The model has the form `Y = g(X) + T * m(X)`, where `Y` is cumulative interconnected direct
operational emissions, `T` is observed workload energy, and `X` contains only `pre_` variables.
The basis coefficients defining `m(X)` are constrained nonnegative. Separate models cover the
workload duration plus 6, 12, and 24 response hours.

Whole workload decisions are split chronologically: the first 70% train the model, a 24-hour
embargo separates periods, and the remaining decisions form the test period. A propensity model
checks whether executed and alternative hours overlap in comparable pre-treatment states. Seven-day
decision blocks are resampled for 80% and 95% uncertainty intervals. For a shift from baseline `a`
to executed time `b`, the point estimate is `energy_mwh * (m(X_a) - m(X_b))`; confidently avoided
emissions are `max(0, lower_80_percent_bound)`.

The estimator never substitutes load forecast error for observed treatment. It remains blocked
unless there are at least 90 completed decisions across 30 days, enough chronological train/test
decisions, acceptable propensity overlap, at least 160 successful block-bootstrap refits, settled
interconnected outcomes, and point-in-time pre-treatment snapshots. Each grid hour retains its
first non-empty operational snapshot, and that snapshot must predate recommendation generation.
These conditions make the effect observationally identified under the
documented no-unmeasured-confounding and consistency assumptions; they do not turn it into a
randomized experiment.

## Double Machine Learning Companion

`make causal-dml` applies the same observed treatment, interconnected outcome, pre-treatment
adjustment set, chronological holdout, and readiness gates. Histogram gradient boosting estimates
the conditional outcome and treatment functions. Rolling-origin cross-fitting generates orthogonal
residuals using only earlier decisions, with the same 24-hour embargo between each training and
validation block.

The final causal stage interacts residual treatment with interpretable regimes for hour, season,
renewable forecast, planned outages, and cross-border congestion. Average effects include
decision-clustered uncertainty intervals; regime summaries expose heterogeneity. Every horizon also
reports its mean-effect difference, effect correlation, and sign disagreement against the original
constrained marginal-response regression. DML is parallel robustness evidence: disagreement keeps
the constrained estimator in place and requires investigation.

For pipeline demonstrations before real workload executions exist, `make causal-synthetic-demo`
creates an isolated, deterministic synthetic portfolio with realistic selection, energy, duration,
grid regimes, and known marginal effects. These artifacts validate computation and presentation
only. They are stored below `reports/demo`, never inserted into Supabase, and are explicitly marked
ineligible for production evidence.

## Validation

Contract generation fails when:

- Either DAG has a cycle or lacks a treatment-to-outcome path.
- The treatment or outcome does not match the grid DAG.
- An adjustment variable is missing, measured after treatment, or is a treatment descendant.
- The paired shift does not conserve energy.
- Workload energy, duration, or response horizons are invalid.
- Baseline precedence differs from planned-start override followed by dashboard-access fallback.

The response is evaluated through workload completion plus a primary six-hour horizon. The 6, 12,
and 24-hour horizons are retained as sensitivity analyses for delayed dispatch, storage, balancing,
and cross-border effects.
