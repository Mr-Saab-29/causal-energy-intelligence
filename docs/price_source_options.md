# France Electricity Spot Price Source Options

## Recommendation

Keep the internal `electricity_prices` contract unchanged and treat the provider as swappable. For this project, the best pragmatic path is:

1. Use **Energy-Charts** as the default source for France day-ahead prices because it does not require a token and returns canonical EUR/MWh time series directly.
2. Keep **ENTSO-E** as the official fallback if we later need the formal transparency-platform source.
3. Use **RTE Services Portal / RTE analyses et données** if its downloadable/API access is easier for the exact French spot series.
4. Use **EPEX SPOT official market data** only if paid/licensed data is acceptable.
5. Use **Ember European wholesale price dataset** if historical country-level prices are enough and API freshness is less important.

## Options

### Energy-Charts API

- Pros: no registration required, France bidding zone supported as `FR`, returns `unix_seconds` and EUR/MWh prices directly.
- Cons: not the primary market operator; attribution/licensing should be preserved in downstream documentation.
- Current adapter: `src/data/sources/energy_charts.py`.
- Current role: default source.

### ENTSO-E Transparency Platform

- Pros: official European transparency source, day-ahead prices in EUR/MWh, France bidding zone supported.
- Cons: token required; XML parsing; occasional access friction.
- Current adapter: `src/data/sources/entsoe.py`.
- Current role: fallback source.

### RTE Price Data

- Pros: France-specific, RTE publishes French and European electricity market price visualizations.
- Cons: API/download mechanics need confirmation; some visual pages may not expose stable backend contracts.
- Fit: good candidate if we can identify a stable CSV/API endpoint.

### EPEX SPOT Market Data

- Pros: primary exchange source for spot market data.
- Cons: official data products are generally paid/licensed.
- Fit: highest-quality source if budget/licensing is available, not ideal for a lightweight academic build.

### Ember Wholesale Price Dataset

- Pros: simple historical European wholesale price dataset.
- Cons: may be aggregated and less suitable for operational hourly workload-shifting simulations if not fresh enough.
- Fit: acceptable fallback for retrospective modeling/reporting.

## Contract Decision

Do not couple downstream code to a provider-specific price format. All price providers should map into:

- `region`
- `timestamp_utc`
- `granularity`
- `market`
- `price_eur_mwh`
- `currency`
- `source`
- `source_record_id`
