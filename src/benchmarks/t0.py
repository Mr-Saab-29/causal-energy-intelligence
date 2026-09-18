"""Day-one dataset freeze and bounded Retrocast inference pilot.

Run with python -m src.benchmarks.t0 prepare|pilot. No production artifacts change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.index = pd.to_datetime(frame.pop('timestamp_utc'), utc=True)
    if frame.index.has_duplicates or not (frame.index == frame.index.floor('h')).all():
        raise ValueError('Expected unique hourly timestamps')
    return frame.sort_index()


def slices(frame: pd.DataFrame, origin: pd.Timestamp, days: int, config: dict):
    history_index = pd.date_range(origin - pd.Timedelta(days=days), periods=days * 24, freq='h')
    future_index = pd.date_range(origin, periods=config['horizon_hours'], freq='h')
    raw = frame.reindex(history_index)
    # Only past observations within this context are used. Never fill evaluation labels.
    history = raw.ffill(limit=config['history_forward_fill_limit_hours'])
    return raw, history, frame.reindex(future_index)


def prepare(config: dict, output: Path) -> dict:
    if output.exists():
        raise ValueError('Snapshot already exists; use a new output directory to preserve it')
    source = Path(config['source'])
    frame = load_frame(source)
    start = pd.Timestamp(config['evaluation_start'])
    end = start + pd.Timedelta(days=config['evaluation_days'])
    columns = config['targets'] + config['weather']
    grid = pd.date_range(start - pd.Timedelta(days=max(config['context_days'])), end, freq='h', inclusive='left')
    frozen = frame.reindex(grid)[columns]
    output.mkdir(parents=True)
    frozen.rename_axis('timestamp_utc').to_csv(output / 'dataset.csv')
    write_json(output / 'config.json', config)
    origins = []
    for origin in pd.date_range(start, periods=config['evaluation_days'], freq='D'):
        for days in config['context_days']:
            raw, history, future = slices(frozen, origin, days, config)
            targets_ok = np.isfinite(history[config['targets']].to_numpy()).all()
            labels_ok = np.isfinite(future[config['targets']].to_numpy()).all()
            origins.append({
                'origin': origin.isoformat(), 'context_days': days,
                'eligible': bool(targets_ok and labels_ok),
                'history_missing_cells': int(raw[config['targets']].isna().sum().sum()),
                'history_unfilled_cells': int(history[config['targets']].isna().sum().sum()),
                'label_missing_cells': int(future[config['targets']].isna().sum().sum()),
                'weather_history_eligible': bool(np.isfinite(history[config['weather']].to_numpy()).all()),
            })
    pd.DataFrame(origins).to_csv(output / 'origins.csv', index=False)
    model_files = {str(p): digest(p) for p in Path('models').glob('*.joblib')}
    windows = {}
    for path in Path('reports/metrics').glob('*baseline_metrics.json'):
        report = json.loads(path.read_text())
        windows[path.name] = list({json.dumps(row['window'], sort_keys=True): row['window']
                                  for row in report.get('metrics', []) if isinstance(row.get('window'), dict)}.values())
    audit = {
        'source_sha256': digest(source), 'dataset_sha256': digest(output / 'dataset.csv'),
        'config_sha256': digest(output / 'config.json'), 'origins_sha256': digest(output / 'origins.csv'),
        'evaluation_start': start.isoformat(), 'evaluation_end_exclusive': end.isoformat(),
        'source_latest': frame.index.max().isoformat(),
        'missing_grid_hours': int((~grid.isin(frame.index)).sum()),
        'missing_cells_by_column': frozen.isna().sum().to_dict(),
        'eligible_origins_by_context': {str(d): sum(r['eligible'] for r in origins if r['context_days'] == d) for d in config['context_days']},
        'model_artifact_sha256': model_files, 'reported_training_windows': windows,
        'baseline_status': config['baseline_status'], 'weather_status': config['weather_status'],
        'limitations': ['Model files have no cryptographically linked training manifests.',
                       'Existing champion selection uses evaluation summaries; this is not an untouched holdout.',
                       'Historical source publication vintages are unavailable: assume latest past hourly values were available.',
                       'Hosted t0-alpha alias cannot be assumed to pin immutable weights.'],
    }
    write_json(output / 'audit.json', audit)
    return audit


def build_request(frame: pd.DataFrame, origin: pd.Timestamp, days: int, config: dict,
                  targets: list[str], mode: str) -> dict:
    if mode not in {'target_only', 'calendar', 'historical_weather', 'related_history'}:
        raise ValueError('Unsupported covariate mode')
    _, history, _ = slices(frame, origin, days, config)
    if not targets or not set(targets) <= set(config['targets']):
        raise ValueError('Unknown targets')
    series = []
    for target in targets:
        if not np.isfinite(history[target].to_numpy()).all():
            raise ValueError('Insufficient finite target history')
        item = {'index': [x.isoformat() for x in history.index], 'target': history[target].tolist()}
        if mode != 'target_only':
            index = pd.date_range(history.index[0], periods=len(history) + config['horizon_hours'], freq='h')
            local = index.tz_convert('Europe/Paris')
            item['future_variables_index'] = [x.isoformat() for x in index]
            item['future_variables'] = {'hour_sin': np.sin(2 * np.pi * local.hour / 24).tolist(),
                                        'hour_cos': np.cos(2 * np.pi * local.hour / 24).tolist(),
                                        'day_of_week': local.dayofweek.tolist()}
        if mode == 'historical_weather':
            if not np.isfinite(history[config['weather']].to_numpy()).all():
                raise ValueError('Insufficient finite historical weather')
            item['hist_variables'] = {name: history[name].tolist() for name in config['weather']}
        if mode == 'related_history':
            related = [name for name in config['targets'] if name != target]
            if not np.isfinite(history[related].to_numpy()).all():
                raise ValueError('Insufficient finite related-series history')
            item['hist_variables'] = {name: history[name].tolist() for name in related}
        series.append(item)
    return {'series': series, 'freq': 'H', 'horizon': config['horizon_hours'],
            'context': days * 24, 'quantiles': config['quantiles']}


def parse_response(body: dict, origin: pd.Timestamp, targets: list[str], config: dict) -> pd.DataFrame:
    expected = pd.date_range(origin, periods=config['horizon_hours'], freq='h')
    groups = body['series']
    if len(groups) != len(targets):
        raise ValueError('Response series count mismatch')
    rows = []
    for target, group in zip(targets, groups):
        if len(group) != 1:
            raise ValueError('Expected exactly one forecast origin')
        result = group[0]
        if not pd.DatetimeIndex(pd.to_datetime(result['index'], utc=True)).equals(expected):
            raise ValueError('Response forecast timestamps mismatch')
        predictions = result['prediction']
        required = ['mean'] + [str(q) for q in config['quantiles']]
        values = np.array([predictions[key] for key in required], dtype=float)
        if values.shape != (len(required), len(expected)) or not np.isfinite(values).all():
            raise ValueError('Invalid prediction lengths or nonfinite values')
        if (np.diff(values[1:], axis=0) < 0).any():
            raise ValueError('Crossed forecast quantiles')
        for step, timestamp in enumerate(expected):
            rows.append({'model': config['model'], 'target': target, 'origin': origin.isoformat(),
                         'timestamp_utc': timestamp.isoformat(), 'lead_hour': step + 1,
                         'prediction': float(predictions['0.5'][step]),
                         **{('mean' if key == 'mean' else f'q{key}'): float(predictions[key][step]) for key in required}})
    return pd.DataFrame(rows)


def pilot(output: Path, live: bool) -> dict:
    config = json.loads((output / 'config.json').read_text())
    audit = json.loads((output / 'audit.json').read_text())
    for name in ['dataset', 'config', 'origins']:
        suffix = 'json' if name == 'config' else 'csv'
        if digest(output / f'{name}.{suffix}') != audit[f'{name}_sha256']:
            raise ValueError(f'Frozen {name} changed')
    frame = load_frame(output / 'dataset.csv')
    origins = pd.read_csv(output / 'origins.csv')
    eligible = origins[origins.eligible].groupby('origin').context_days.nunique()
    common = list(eligible[eligible == len(config['context_days'])].index)
    if not common:
        raise ValueError('No common eligible origins across all context lengths')
    # Three distinct dates, four requests: demand then multi-target/context/covariate probes.
    dates = [common[i] for i in sorted(set([0, len(common)//2, len(common)-1]))]
    cases = [(dates[0], 7, config['targets'][:1], 'target_only'),
             (dates[len(dates)//2], 30, config['targets'], 'calendar'),
             (dates[-1], 90, config['targets'], 'calendar')]
    weather = origins[(origins.context_days == 30) & origins.eligible & origins.weather_history_eligible]
    if not weather.empty:
        cases.append((weather.iloc[-1].origin, 30, config['targets'][:1], 'historical_weather'))
    key = os.getenv('TFC_API_KEY')
    report = {'status': 'dry_run', 'api_key_configured': bool(key), 'cases': [],
              'model_version': 'hosted alias; immutable revision not exposed',
              'covariate_requests_accepted_live': False, 'joint_multivariate_verified': False}
    for date, days, targets, mode in cases:
        origin = pd.Timestamp(date)
        payload = build_request(frame, origin, days, config, targets, mode)
        identity = {'model': config['model'], 'endpoint': config['api_url'], 'payload': payload}
        request_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        request_path = output / f'{request_id}.request.json'
        response_path = output / f'{request_id}.response.json'
        write_json(request_path, identity)
        case = {'request_id': request_id, 'origin': date, 'context_days': days, 'targets': targets,
                'mode': mode, 'request_bytes': request_path.stat().st_size}
        report['cases'].append(case)
        if not live:
            continue
        if response_path.exists():
            body = json.loads(response_path.read_text())
            case['cached'] = True
        elif not key:
            report['status'] = 'blocked_missing_api_key'
            break
        else:
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=180) as client:
                    response = client.post(config['api_url'], params={'model': config['model']},
                                           headers={'Authorization': f'Bearer {key}'}, json=payload)
                case['elapsed_seconds'] = time.perf_counter() - start
                case['http_status'] = response.status_code
                response.raise_for_status()
                body = response.json()
                parse_response(body, origin, targets, config)
                write_json(response_path, body)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                # Do not log response bodies or headers, which can contain user data or secrets.
                case['error_type'] = type(error).__name__
                report['status'] = 'failed'
                break
        predictions = parse_response(body, origin, targets, config)
        predictions['context_days'] = days
        predictions['covariate_mode'] = mode
        predictions.to_csv(output / f'{request_id}.predictions.csv', index=False)
        case['rows'] = len(predictions)
    else:
        if live:
            report['status'] = 'completed'
            report['covariate_requests_accepted_live'] = True
    write_json(output / ('pilot.json' if live else 'pilot_plan.json'), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'pilot'])
    parser.add_argument('--config', type=Path, default=Path('config/benchmark_t0.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--live', action='store_true', help='Send at most four pilot requests; may consume API quota')
    args = parser.parse_args()
    load_dotenv()
    config = json.loads(args.config.read_text())
    output = args.output or Path('reports/benchmarks') / config['benchmark_id']
    result = prepare(config, output) if args.action == 'prepare' else pilot(output, args.live)
    print(json.dumps(result, indent=2))
    if result.get('status') in {'failed', 'blocked_missing_api_key'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
