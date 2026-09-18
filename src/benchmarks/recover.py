"""Recover observed benchmark cells from canonical sources, without changing production data."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from src.benchmarks.t0 import digest, load_frame, prepare, write_json
from src.data.sources.odre import aggregate_odre_to_hourly_mwh

ROOT = Path('reports/benchmarks')
RECOVERY = ROOT / 't0_recovery_v2'
SNAPSHOT = Path(os.environ.get('BENCHMARK_SNAPSHOT', str(ROOT / 't0_2026_05_26_to_08_23_v3')))


def fill_observed(frame: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing cells only, retaining all existing values and timestamp positions."""
    if observed.index.has_duplicates:
        raise ValueError('Duplicate observation timestamps')
    aligned = observed.reindex(index=frame.index, columns=frame.columns)
    mask = frame.isna() & aligned.notna()
    return frame.where(~mask, aligned), int(mask.sum().sum())


def main() -> None:
    raise ValueError('Legacy recovery uses uncorrected targets. Run python -m src.benchmarks.rebuild_energy instead.')


def legacy_recovery() -> None:
    if RECOVERY.exists() or SNAPSHOT.exists():
        raise ValueError('Recovery already exists; preserve it and use a new version')
    config = json.loads(Path('config/benchmark_t0.json').read_text())
    paths = [Path(config['source']), Path('data/processed/hourly_electricity_mix.csv'),
             Path('data/processed/weather_observations.csv')]
    hashes = {str(p): digest(p) for p in paths}
    original = load_frame(paths[0])[config['targets'] + config['weather']]
    end = pd.Timestamp(config['evaluation_start']) + pd.Timedelta(days=config['evaluation_days'])
    frame = original.reindex(pd.date_range(original.index.min(), end, inclusive='left', freq='h'))
    mix = pd.read_csv(paths[1])
    mix = mix[(mix.region == 'FR') & (mix.scope == 'national')].copy()
    mix.index = pd.to_datetime(mix.timestamp_utc, utc=True)
    frame, target_cells = fill_observed(frame, mix[config['targets']])
    mapping = {'temperature_c': 'avg_temperature_c', 'wind_speed_mps': 'avg_wind_speed_mps',
               'shortwave_radiation_wm2': 'avg_shortwave_radiation_wm2'}
    weather = pd.read_csv(paths[2], usecols=['timestamp_utc', 'region', *mapping])
    weather.timestamp_utc = pd.to_datetime(weather.timestamp_utc, utc=True)
    if weather.duplicated(['timestamp_utc', 'region']).any():
        raise ValueError('Duplicate region weather observations')
    # Only complete twelve-region aggregates may repair missing model-table cells.
    grouped = weather.groupby('timestamp_utc')
    aggregated = grouped[list(mapping)].mean().rename(columns=mapping)
    complete = grouped[list(mapping)].count().eq(12).all(axis=1) & grouped.region.nunique().eq(12)
    aggregated = aggregated.where(complete, axis=0)
    frame, weather_cells = fill_observed(frame, aggregated)
    RECOVERY.mkdir(parents=True)
    start_grid = pd.Timestamp(config['evaluation_start']) - pd.Timedelta(days=90)
    gap_index = frame.loc[start_grid:, config['targets']].index[frame.loc[start_grid:, config['targets']].isna().any(axis=1)]
    attempts = []
    for day in sorted(set(gap_index.date)):
        for dataset in ['eco2mix-national-cons-def', 'eco2mix-national-tr']:
            url = f'https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/records'
            start = pd.Timestamp(day, tz='UTC')
            params = {'limit': 100, 'order_by': 'date_heure',
                      'where': f"date_heure >= date'{start.isoformat()}' AND date_heure < date'{(start + pd.Timedelta(days=1)).isoformat()}'"}
            entry = {'url': url, 'params': params}
            attempts.append(entry)
            try:
                response = httpx.get(url, params=params, timeout=30)
                entry['http_status'] = response.status_code
                response.raise_for_status()
                payload = response.json()
                if payload.get('total_count', 0) > 100:
                    raise ValueError('Refuse truncated source day')
                write_json(RECOVERY / f'{day}-{dataset}.json', payload)
                observations = aggregate_odre_to_hourly_mwh(payload.get('results', []), scope='national', dataset=dataset)
                if observations:
                    observed = pd.DataFrame([{'timestamp_utc': x.timestamp_utc,
                                               **{col: float(getattr(x, col)) if getattr(x, col) is not None else None
                                                  for col in config['targets']}} for x in observations])
                    observed.index = pd.to_datetime(observed.pop('timestamp_utc'), utc=True)
                    frame, count = fill_observed(frame, observed)
                    entry['recovered_cells'] = count
                else:
                    entry['recovered_cells'] = 0
            except (httpx.HTTPError, ValueError) as error:
                entry['error_type'] = type(error).__name__
    output = RECOVERY / 'observed_source.csv'
    frame.rename_axis('timestamp_utc').to_csv(output)
    config.update(benchmark_id=SNAPSHOT.name, source=str(output),
                  weather_status='historical_observations_only',
                  baseline_status='isolated_fixed_algorithm_training_required')
    audit = prepare(config, SNAPSHOT)
    # List every filled observation, including its original missing status.
    before = original.reindex(frame.index)
    repaired = before.isna() & frame.notna()
    rows, cols = repaired.to_numpy().nonzero()
    pd.DataFrame({'timestamp_utc': frame.index[rows], 'column': frame.columns[cols],
                  'value': frame.to_numpy()[rows, cols]}).to_csv(RECOVERY / 'recovered_cells.csv', index=False)
    report = {'created_at_utc': datetime.now(UTC).isoformat(), 'input_sha256': hashes,
              'output_sha256': digest(output), 'local_target_cells_recovered': target_cells,
              'local_weather_cells_recovered': weather_cells, 'upstream_attempts': attempts,
              'eligible_origins_by_context': audit['eligible_origins_by_context'],
              'remaining_missing_cells': audit['missing_cells_by_column'],
              'weather_policy': 'historical only; no future observed weather',
              'production_sources_unchanged': all(digest(p) == hashes[str(p)] for p in paths)}
    write_json(RECOVERY / 'recovery.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
