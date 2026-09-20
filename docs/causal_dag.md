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
constraints. It is documented now but is not estimated by the current pipeline.

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
