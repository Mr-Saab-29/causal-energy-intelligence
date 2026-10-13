# Structural Validation

The statistical estimators and a dispatch model answer the same marginal question through
different assumptions. The constrained regression and DML learn from workload decisions and
observed grid states. A PyPSA-Eur counterfactual instead fixes a European network and asks how
optimized dispatch changes when France receives an additional workload. Agreement is useful
corroboration. Disagreement is model-risk evidence; the pipeline does not average the estimates.

## Current modes

`make causal-structural-validation` selects inputs in this order:

1. Paired PyPSA-Eur NetCDF networks listed in
   `data/external/pypsa_eur_counterfactual_manifest.json`.
2. Precomputed dispatch responses in
   `data/external/structural_dispatch_counterfactuals.csv`.
3. The synthetic zonal merit-order fallback used by the personal-project demonstration.

Only mode 1 can become eligible for a real-world structural causal claim. The synthetic fallback
is useful for exercising the complete product and model-risk gate, but it is not evidence about
the actual European grid.

## PyPSA-Eur integration

PyPSA-Eur writes solved networks to `results/{run}/networks/solved_{horizon}.nc` and
supports dispatch-only re-solves with fixed capacities. Build paired baseline and intervention
operations networks with identical capacities, constraints, weather, fuel prices, and snapshots.
In the intervention run, add the workload to France; do not change any other input.

Create the manifest:

```json
{
  "runs": [
    {
      "run_id": "fr-2026-01-15T12:00-6h",
      "start_utc": "2026-01-15T12:00:00Z",
      "horizon_hours": 6,
      "perturbation_mwh": 1.0,
      "baseline_network": "pypsa/baseline_6h.nc",
      "counterfactual_network": "pypsa/intervention_6h.nc"
    }
  ]
}
```

Paths may be absolute or relative to the manifest. Install PyPSA in the environment used for
this command; it remains optional for the normal forecasting and dashboard environment.

The importer computes:

```text
structural response =
  (counterfactual direct emissions - baseline direct emissions) / workload MWh
```

Generator output is converted using snapshot hours, carrier-to-source aliases, and the
output-based factors in `config/emission_factors.yaml`. Dispatch mapping coverage must be at
least 95%.

## Model-risk policy

- `low`: sign agrees and relative difference is at most 35% for every estimator/horizon
  comparison.
- `elevated`: sign agrees but magnitude differs beyond 35%, or dispatch coverage is
  insufficient.
- `high`: the structural and statistical estimates disagree in sign.

Any elevated or high result produces `do_not_treat_as_structurally_validated`. Statistical
and structural estimates remain visible separately for investigation.
