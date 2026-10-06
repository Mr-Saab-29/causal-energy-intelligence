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
