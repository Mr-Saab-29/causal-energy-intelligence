"""Bounded paired input-sensitivity diagnostic for hosted t0-alpha (seven requests)."""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx
import numpy as np
import pandas as pd

from src.benchmarks.recover import SNAPSHOT
from src.benchmarks.rebuild_energy import verify_energy_source
from src.benchmarks.t0 import build_request, digest, load_frame, parse_response, slices, write_json


def diagnostic_requests(frame: pd.DataFrame, origin: pd.Timestamp, config: dict) -> dict:
    target = config['targets'][0]
    target_only = build_request(frame, origin, 30, config, [target], 'target_only')
    calendar = build_request(frame, origin, 30, config, [target], 'calendar')
    weather = build_request(frame, origin, 30, config, [target], 'historical_weather')
    weather_shuffled = deepcopy(weather)
    permutation = np.random.default_rng(42).permutation(30 * 24)
    for name, values in weather_shuffled['series'][0]['hist_variables'].items():
        weather_shuffled['series'][0]['hist_variables'][name] = np.asarray(values)[permutation].tolist()
    related = deepcopy(calendar)
    _, history, _ = slices(frame, origin, 30, config)
    columns = [name for name in config['targets'] if name != target]
    if not np.isfinite(history[columns].to_numpy()).all():
        raise ValueError('Missing related-series history')
    related['series'][0]['hist_variables'] = {name: history[name].tolist() for name in columns}
    related_shuffled = deepcopy(related)
    for name, values in related_shuffled['series'][0]['hist_variables'].items():
        related_shuffled['series'][0]['hist_variables'][name] = np.asarray(values)[permutation].tolist()
    return {'target_only': target_only, 'target_only_repeat': deepcopy(target_only),
            'calendar': calendar, 'historical_weather': weather, 'weather_shuffled': weather_shuffled,
            'related_history': related, 'related_shuffled': related_shuffled}


def compare_predictions(predictions: dict[str, np.ndarray]) -> dict:
    repeat = float(np.max(np.abs(predictions['target_only'] - predictions['target_only_repeat'])))
    threshold = max(1e-6, repeat * 5)
    comparisons = {}
    for name, left, right in [('calendar', 'target_only', 'calendar'),
                              ('historical_weather', 'calendar', 'historical_weather'),
                              ('weather_alignment', 'historical_weather', 'weather_shuffled'),
                              ('related_history', 'calendar', 'related_history'),
                              ('related_alignment', 'related_history', 'related_shuffled')]:
        delta = float(np.max(np.abs(predictions[left] - predictions[right])))
        comparisons[name] = {'max_absolute_median_difference': delta,
                             'observed_sensitivity_above_repeat_tolerance': delta > threshold}
    return {'repeat_max_absolute_difference': repeat, 'comparison_tolerance': threshold,
            'comparisons': comparisons,
            'interpretation': 'Input sensitivity on one pre-evaluation origin, not accuracy benefit or proof of joint attention.'}


def run(snapshot: Path = SNAPSHOT) -> dict:
    load_dotenv()
    key = os.getenv('TFC_API_KEY')
    if not key:
        raise ValueError('Configure TFC_API_KEY in local .env; do not paste it into logs')
    config = json.loads((snapshot / 'config.json').read_text())
    verify_energy_source(config)
    audit = json.loads((snapshot / 'audit.json').read_text())
    if digest(snapshot / 'dataset.csv') != audit['dataset_sha256'] or digest(snapshot / 'config.json') != audit['config_sha256']:
        raise ValueError('Frozen snapshot changed')
    frame = load_frame(snapshot / 'dataset.csv')
    origin = pd.Timestamp(config['evaluation_start']) - pd.Timedelta(days=1)
    requests = diagnostic_requests(frame, origin, config)
    output = snapshot / 'covariate_diagnostic'
    output.mkdir(exist_ok=True)
    predictions, cases = {}, []
    for name, payload in requests.items():
        identity = {'endpoint': config['api_url'], 'model': config['model'], 'case': name, 'payload': payload}
        request_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = output / f'{name}.response.json'
        metadata_path = output / f'{name}.metadata.json'
        write_json(output / f'{name}.request.json', identity)
        if path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata['request_sha256'] != request_hash or metadata['response_sha256'] != digest(path):
                raise ValueError('Diagnostic cache changed')
            body = json.loads(path.read_text())
        else:
            started = time.perf_counter()
            response = httpx.post(config['api_url'], params={'model': config['model']},
                                  headers={'Authorization': f'Bearer {key}'}, json=payload, timeout=180)
            if response.status_code != 200:
                raise RuntimeError(f'Diagnostic {name} stopped: HTTP {response.status_code}; no automatic retry')
            body = response.json()
            parse_response(body, origin, [config['targets'][0]], config)
            write_json(path, body)
            metadata = {'request_sha256': request_hash, 'response_sha256': digest(path),
                        'run_at_utc': datetime.now(UTC).isoformat(), 'elapsed_seconds': time.perf_counter() - started}
            write_json(metadata_path, metadata)
        prediction = parse_response(body, origin, [config['targets'][0]], config)
        prediction.to_csv(output / f'{name}.predictions.csv', index=False)
        predictions[name] = prediction.prediction.to_numpy()
        cases.append({'name': name, **metadata})
    report = {'status': 'completed', 'origin': origin.isoformat(), 'target': config['targets'][0],
              'context_days': 30, 'cases': cases, **compare_predictions(predictions),
              'joint_multivariate_claim': False, 'future_weather_used': False}
    write_json(output / 'summary.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
