"""Verify Day 1 evidence and write a readiness report without network calls."""
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.benchmarks.recover import SNAPSHOT
from src.benchmarks.rebuild_energy import verify_energy_source
from src.benchmarks.t0 import digest, write_json


def validate(snapshot: Path = SNAPSHOT) -> dict:
    config = json.loads((snapshot / 'config.json').read_text())
    verify_energy_source(config)
    audit = json.loads((snapshot / 'audit.json').read_text())
    for name, suffix in [('dataset', 'csv'), ('config', 'json'), ('origins', 'csv')]:
        if digest(snapshot / f'{name}.{suffix}') != audit[f'{name}_sha256']:
            raise ValueError(f'Changed snapshot {name}')
    if digest(Path(config['source'])) != audit['source_sha256']:
        raise ValueError('Changed training source')
    cutoff = pd.Timestamp(config['evaluation_start'])
    baselines = snapshot / 'baselines'
    manifests = list(baselines.glob('*.manifest.json'))
    if len(manifests) != len(config['targets']) * 4:
        raise ValueError('Incomplete baseline manifest set')
    expected = {(algorithm, target, mode) for algorithm in ['ridge', 'lightgbm']
                for target in config['targets'] for mode in ['calendar', 'historical_weather']}
    found = set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest.get('benchmark_source_sha256') != audit['source_sha256']:
            raise ValueError('Baseline was not trained from this corrected benchmark source')
        algorithm, target, mode = (manifest[k] for k in ['algorithm', 'target', 'mode'])
        found.add((algorithm, target, mode))
        artifact = baselines / f'{algorithm}-{target}-{mode}.joblib'
        training = baselines / f'{target}-{mode}.training.csv.gz'
        if digest(artifact) != manifest['artifact_sha256'] or digest(training) != manifest['training_data_sha256']:
            raise ValueError('Changed baseline training data or artifact')
        if digest(baselines / 'training_source.csv') != manifest['training_source_sha256']:
            raise ValueError('Changed baseline source')
        if pd.Timestamp(manifest['label_end']) >= cutoff:
            raise ValueError('Baseline trained on evaluation labels')
        if digest(Path('src/benchmarks/baselines.py')) != manifest['implementation_sha256']:
            raise ValueError('Baseline implementation differs from training manifest')
    if found != expected:
        raise ValueError('Unexpected baseline manifest set')
    origins = pd.read_csv(snapshot / 'origins.csv')
    counts = origins.groupby('context_days')[['eligible', 'weather_history_eligible']].sum()
    if set(counts.index) != set(config['context_days']) or not counts.eq(config['evaluation_days']).all().all():
        raise ValueError('Incomplete matched target/weather origins')
    for name, expected_hash in audit['model_artifact_sha256'].items():
        if digest(Path(name)) != expected_hash:
            raise ValueError('Production model changed since snapshot; review separately')
    diagnostic = json.loads((snapshot / 'covariate_diagnostic/summary.json').read_text())
    if diagnostic['status'] != 'completed' or len(diagnostic['cases']) != 7:
        raise ValueError('Incomplete covariate diagnostic')
    if not all(row['observed_sensitivity_above_repeat_tolerance'] for row in diagnostic['comparisons'].values()):
        raise ValueError('Covariate influence remains inconclusive')
    for case in diagnostic['cases']:
        path = snapshot / 'covariate_diagnostic' / f"{case['name']}.response.json"
        if digest(path) != case['response_sha256']:
            raise ValueError('Changed diagnostic response')
    recovery = json.loads(Path(config['energy_rebuild_report']).read_text())
    report = {'status': 'ready_for_day_2', 'verified_at_utc': datetime.now(UTC).isoformat(),
              'snapshot': str(snapshot), 'verified_baseline_artifacts': len(manifests),
              'training_cutoff_exclusive': cutoff.isoformat(),
              'eligible_origins': {str(k): {name: int(value) for name, value in row.items()} for k, row in counts.to_dict('index').items()},
              'weather_scope': 'historical observed covariates only; calendar known in future',
              'covariate_input_sensitivity': diagnostic['comparisons'],
              'source_recovery_production_unchanged': recovery['production_sources_unchanged'],
              'remaining_disclosures': ['Hosted t0-alpha checkpoint revision not pinned.',
                                        'Pretraining overlap with public data is unknown.',
                                        'Energy/weather publication vintages unverified; retrospective availability assumed.',
                                        'Related histories are explicit covariates; joint multivariate attention not established.',
                                        'Input sensitivity is not evidence of better accuracy.',
                                        'Baselines are newly trained benchmark variants, not the original production checkpoints.']}
    write_json(snapshot / 'readiness.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
