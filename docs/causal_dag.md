# Causal DAG

Initial causal graph assumptions:

```text
weather -> electricity_demand
electricity_demand -> electricity_price
renewable_generation -> carbon_intensity
carbon_intensity -> workload_shift_decision
electricity_price -> workload_shift_decision
```

## Key Variables

- `weather`: temperature, wind, solar irradiance, and seasonal effects.
- `electricity_demand`: aggregate regional load.
- `renewable_generation`: wind, solar, hydro, and other low-carbon supply.
- `electricity_price`: market price signal.
- `carbon_intensity`: grid emissions intensity.
- `workload_shift_decision`: recommended workload timing intervention.

## Next Work

- Validate confounders and adjustment sets.
- Define treatment variables for workload shifting.
- Define outcomes for cost, carbon, and service-level impact.
- Compare observational estimates against simulated counterfactuals.

