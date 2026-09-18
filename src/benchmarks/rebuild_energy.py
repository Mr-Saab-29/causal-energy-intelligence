"""Rebuild national benchmark energy from source readings into a new snapshot.

Run manually with python -m src.benchmarks.rebuild_energy. Fetches public ODRE
data only; no model training, t0 calls, database writes or production replacement.
Completed daily downloads are verified and reused after interruption.
"""
from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.recover import RECOVERY, SNAPSHOT
from src.benchmarks.t0 import digest, load_frame, prepare, slices, write_json
from src.data.source_config import ODRE_NATIONAL_DATASET, ODRE_NATIONAL_HISTORICAL_DATASET
from src.data.sources.odre import aggregate_odre_to_hourly_mwh, fetch_odre_records

AGGREGATION_VERSION = 'source_interval_v2'


def verify_energy_source(config: dict) -> None:
    """Refuse legacy data even if its earlier completeness checks passed."""
    if config.get('energy_aggregation_version') != AGGREGATION_VERSION:
        raise ValueError('Uncorrected energy snapshot: run src.benchmarks.rebuild_energy in a new directory')
    evidence = Path(config['energy_rebuild_report'])
    if digest(evidence) != config['energy_rebuild_report_sha256']:
        raise ValueError('Energy rebuild evidence changed')
    report = json.loads(evidence.read_text())
    if (report['status'] != 'completed' or report.get('aggregation_version') != AGGREGATION_VERSION
            or digest(Path(config['source'])) != report['output_sha256']):
        raise ValueError('Corrected energy source changed or rebuild incomplete')
    if digest(Path('src/data/sources/odre.py')) != report['aggregation_implementation_sha256']:
        raise ValueError('Energy aggregation implementation changed since rebuild')


def source_day(day: pd.Timestamp, dataset: str, cache: Path) -> tuple[list[dict], dict]:
    path = cache / f'{day.date()}-{dataset}.json'
    metadata_path = path.with_suffix('.metadata.json')
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['sha256'] != digest(path) or metadata['dataset'] != dataset or metadata['day'] != str(day.date()):
            raise ValueError(f'Raw source cache changed: {path}')
        records = json.loads(path.read_text())
    else:
        records = list(fetch_odre_records(dataset, day.date(), day.date()))
        # Preserve raw duplicates as evidence. The aggregator checks whether
        # measurements agree before counting a repeated UTC instant once.
        write_json(path, records)
        metadata = {'dataset': dataset, 'day': str(day.date()), 'sha256': digest(path),
                    'retrieved_at_utc': datetime.now(UTC).isoformat(), 'rows': len(records)}
        write_json(metadata_path, metadata)
    timestamps = set()
    for record in records:
        timestamp = pd.Timestamp(record['date_heure'])
        if timestamp.tzinfo is None or not day <= timestamp < day + pd.Timedelta(days=1):
            raise ValueError(f'Source row outside requested UTC day: {path}')
        if timestamp.minute % 15 or timestamp.second or timestamp.microsecond:
            raise ValueError(f'Source row outside quarter-hour grid: {path}')
        timestamps.add(timestamp)
    if len(timestamps) > 96:
        raise ValueError(f'Excess unique source timestamps: {day.date()} / {dataset}')
    metadata = {**metadata, 'unique_timestamps': len(timestamps),
                'duplicate_timestamp_rows': len(records) - len(timestamps)}
    return records, {'path': str(path), **metadata}


def rebuild() -> dict:
    if SNAPSHOT.exists():
        raise ValueError(f'Preserve existing snapshot {SNAPSHOT}; choose a new BENCHMARK_SNAPSHOT')
    output = SNAPSHOT.with_name(SNAPSHOT.name + '_energy_rebuild')
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'raw'
    cache.mkdir(exist_ok=True)
    config = json.loads(Path('config/benchmark_t0.json').read_text())
    # Reuse weather only. Every energy target is discarded and reconstructed.
    previous = RECOVERY / 'observed_source.csv'
    original = load_frame(previous)
    end = pd.Timestamp(config['evaluation_start']) + pd.Timedelta(days=config['evaluation_days'])
    index = pd.date_range(original.index.min().floor('D'), end, freq='h', inclusive='left')
    frame = original.reindex(index)[config['targets'] + config['weather']].copy()
    frame[config['targets']] = np.nan
    provenance = pd.Series(index=index, dtype='object', name='dataset')
    inputs, daily = [], []
    for number, day in enumerate(pd.date_range(index.min(), index.max().floor('D'), freq='D'), 1):
        day_index = pd.date_range(day, periods=24, freq='h')
        for dataset in [ODRE_NATIONAL_HISTORICAL_DATASET, ODRE_NATIONAL_DATASET]:
            records, metadata = source_day(day, dataset, cache)
            inputs.append(metadata)
            observations = aggregate_odre_to_hourly_mwh(records, scope='national', dataset=dataset)
            for observation in observations:
                timestamp = pd.Timestamp(observation.timestamp_utc)
                values = [getattr(observation, target) for target in config['targets']]
                # Choose a complete hourly mix from one source; do not combine
                # historical and real-time components or reuse old target cells.
                if pd.isna(provenance.loc[timestamp]):
                    if all(value is not None and value.is_finite() for value in values):
                        frame.loc[timestamp, config['targets']] = [float(value) for value in values]
                        provenance.loc[timestamp] = dataset
            if provenance.loc[day_index].notna().all():
                break
        complete = int(provenance.loc[day_index].notna().sum())
        daily.append({'date': str(day.date()), 'complete_hours': complete})
        if number % 30 == 0:
            print(f'Rebuilt {number} source days through {day.date()}', flush=True)

    source = output / 'observed_source.csv'
    frame.rename_axis('timestamp_utc').to_csv(source)
    provenance.rename_axis('timestamp_utc').to_csv(output / 'hourly_provenance.csv')
    pd.DataFrame(daily).to_csv(output / 'daily_coverage.csv', index=False)
    eligible = {}
    for days in config['context_days']:
        eligible[str(days)] = 0
        for origin in pd.date_range(config['evaluation_start'], periods=config['evaluation_days'], freq='D'):
            _, history, future = slices(frame, origin, days, config)
            if (np.isfinite(history[config['targets'] + config['weather']].to_numpy()).all()
                    and np.isfinite(future[config['targets']].to_numpy()).all()):
                eligible[str(days)] += 1
    complete = all(count == config['evaluation_days'] for count in eligible.values())
    report = {'status': 'completed' if complete else 'blocked_missing_observations',
              'created_at_utc': datetime.now(UTC).isoformat(), 'aggregation_version': AGGREGATION_VERSION,
              'aggregation_implementation_sha256': digest(Path('src/data/sources/odre.py')),
              'weather_source': str(previous), 'weather_source_sha256': digest(previous),
              'output_sha256': digest(source), 'upstream_inputs': inputs,
              'hourly_provenance_sha256': digest(output / 'hourly_provenance.csv'),
              'eligible_origins_by_context': eligible,
              'missing_target_hours': int(frame[config['targets']].isna().any(axis=1).sum()),
              'policy': 'Latest available consolidated/definitive complete hourly mix; real-time fallback. No target imputation in saved source.',
              'production_sources_unchanged': True}
    report_path = output / 'rebuild.json'
    write_json(report_path, report)
    if not complete:
        raise ValueError(f'Incomplete matched origins: {eligible}. Inspect {output}/daily_coverage.csv; no snapshot frozen. Do not fill evaluation labels.')
    config.update(benchmark_id=SNAPSHOT.name, source=str(source),
                  energy_aggregation_version=AGGREGATION_VERSION,
                  energy_rebuild_report=str(report_path), energy_rebuild_report_sha256=digest(report_path),
                  baseline_status='isolated_fixed_algorithm_training_required',
                  weather_status='historical_observations_only')
    prepare(config, SNAPSHOT)
    return {'status': 'completed', 'snapshot': str(SNAPSHOT), 'source': str(source),
            'eligible_origins_by_context': eligible, 'next_step': 'Train new baselines; do not copy v2 model or forecast caches.'}


if __name__ == '__main__':
    print(json.dumps(rebuild(), indent=2))
