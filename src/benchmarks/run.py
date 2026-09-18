"""Frozen, resumable t0 benchmark. prepare, local, remote --live, then report."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx
import joblib
import numpy as np
import pandas as pd

from src.benchmarks.baselines import origin_features
from src.benchmarks.recover import SNAPSHOT
from src.benchmarks.rebuild_energy import verify_energy_source
from src.benchmarks.t0 import build_request, digest, load_frame, parse_response, write_json
from src.benchmarks.validate import validate
from src.carbon.intensity import load_emission_factor_config

RUN = SNAPSHOT / 'experiments_v1'
VARIANTS = [
    {'id': 'calendar_30d', 'mode': 'calendar', 'days': 30, 'stress': None},
    {'id': 'target_30d', 'mode': 'target_only', 'days': 30, 'stress': None},
    {'id': 'weather_30d', 'mode': 'historical_weather', 'days': 30, 'stress': None},
    {'id': 'related_30d', 'mode': 'related_history', 'days': 30, 'stress': None},
    {'id': 'calendar_7d', 'mode': 'calendar', 'days': 7, 'stress': None},
    {'id': 'calendar_90d', 'mode': 'calendar', 'days': 90, 'stress': None},
    *[{'id': name, 'mode': 'historical_weather', 'days': 30, 'stress': name}
      for name in ['dropout_10', 'dropout_30', 'outage_6h', 'noise_10', 'noise_30']],
]


def prepare_run() -> dict:
    validate()
    if RUN.exists():
        raise ValueError('Run already prepared; use local/remote/report to resume')
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    data = load_frame(Path(config['source']))
    cutoff = pd.Timestamp(config['evaluation_start'])
    pre = data.loc[data.index < cutoff]
    scales = {}
    for target in config['targets']:
        series = pre[target].reindex(pd.date_range(pre.index.min(), pre.index.max(), freq='h'))
        value = float((series - series.shift(168)).abs().mean())
        scales[target] = value if np.isfinite(value) and value > 0 else None
    thresholds = {column: {'p05': float(pre[column].quantile(.05)), 'p95': float(pre[column].quantile(.95))}
                  for column in ['consumption_mwh', 'total_production_mwh', 'wind_mwh', 'avg_temperature_c']}
    price_path = Path('data/processed/electricity_prices.csv')
    prices = pd.read_csv(price_path)
    prices['timestamp_utc'] = pd.to_datetime(prices.timestamp_utc, utc=True)
    if 'region' in prices:
        prices = prices[prices.region == 'FR']
    if prices.timestamp_utc.duplicated().any():
        raise ValueError('Duplicate price timestamps need explicit provenance resolution')
    index = pd.date_range(cutoff - pd.Timedelta(days=1), periods=91*24, freq='h')
    prices = prices.set_index('timestamp_utc').reindex(index)[['price_eur_mwh']]
    recovery_path = Path('reports/benchmarks/e8_price_recovery.json')
    if recovery_path.exists():
        recovered = pd.DataFrame(json.loads(recovery_path.read_text())['rows'])
        recovered.index = pd.to_datetime(recovered.pop('timestamp_utc'), utc=True)
        prices = prices.fillna(recovered.reindex(prices.index))
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError('Incomplete prices for fixed E8 reference')
    RUN.mkdir()
    prices.rename_axis('timestamp_utc').to_csv(RUN / 'prices.csv')
    protocol = {'created_at_utc': datetime.now(UTC).isoformat(), 'config': config, 'variants': VARIANTS,
                'snapshot_sha256': digest(SNAPSHOT / 'dataset.csv'),
                'config_sha256': digest(SNAPSHOT / 'config.json'),
                'origins_sha256': digest(SNAPSHOT / 'origins.csv'),
                'price_source_sha256': digest(price_path), 'price_recovery_sha256': digest(recovery_path) if recovery_path.exists() else None, 'prices_sha256': digest(RUN / 'prices.csv'),
                'emission_factors': load_emission_factor_config('config/emission_factors.yaml')['direct_operational_emissions'],
                'mase_scales': scales, 'regime_thresholds': thresholds,
                'weather_statistics': {name: {'mean': float(pre[name].mean()), 'std': float(pre[name].std())} for name in config['weather']},
                'seed': 42, 'expected_api_requests': 90 * len(VARIANTS),
                'decision_policy': {'duration_hours': 1, 'max_delay_hours': 23, 'carbon_weight': .8,
                                    'price_weight': .2, 'price_forecast': 'previous-day same-hour price',
                                    'energy_mwh': 1, 'methodology': 'direct_operational_emissions',
                                    'postprocessing': 'clip negative generation to zero for carbon accounting only',
                                    'ranking': 'weighted normalized ordinal ranks; ties by timestamp; no learned overlay'},
                'stress_policy': 'Perturb historical weather only; forward fill at most 24h then use pre-period mean; never future weather.',
                'baseline_policy': 'Fixed seven-day features; E4 context variants and untrained E2 input configurations are N/A.',
                'implementation_sha256': digest(Path(__file__))}
    write_json(RUN / 'protocol.json', protocol)
    write_json(RUN / 'status.json', {'status': 'prepared', 'completed': 0, 'expected': protocol['expected_api_requests']})
    return protocol


def load_run():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    verify_energy_source(protocol['config'])
    checks = [(SNAPSHOT/'dataset.csv', 'snapshot_sha256'), (SNAPSHOT/'config.json', 'config_sha256'),
              (SNAPSHOT/'origins.csv', 'origins_sha256'), (RUN/'prices.csv', 'prices_sha256')]
    for path, key in checks:
        if digest(path) != protocol[key]:
            raise ValueError(f'Frozen input changed: {path.name}')
    return protocol, load_frame(SNAPSHOT / 'dataset.csv')


def stress_history(frame: pd.DataFrame, origin: pd.Timestamp, stress: str | None, protocol: dict) -> pd.DataFrame:
    if stress is None:
        return frame
    output = frame.copy()
    index = pd.date_range(origin - pd.Timedelta(days=30), periods=720, freq='h')
    config = protocol['config']
    seed = int(hashlib.sha256(f"{origin.isoformat()}:{protocol['seed']}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for column in config['weather']:
        values = output.reindex(index)[column].copy()
        if stress.startswith('dropout_'):
            rate = int(stress.split('_')[1]) / 100
            values.loc[rng.random(len(values)) < rate] = np.nan
        elif stress == 'outage_6h':
            values.iloc[-6:] = np.nan
        elif stress.startswith('noise_'):
            sigma = int(stress.split('_')[1]) / 100 * protocol['weather_statistics'][column]['std']
            values += rng.normal(0, sigma, len(values))
        else:
            raise ValueError('Unknown stress')
        values = values.ffill(limit=24).fillna(protocol['weather_statistics'][column]['mean'])
        output.loc[index, column] = values
    return output


def origins(protocol: dict):
    config = protocol['config']
    return pd.date_range(config['evaluation_start'], periods=config['evaluation_days'], freq='D')


def annotate(predictions: pd.DataFrame, variant: str, elapsed: float | None = None):
    predictions['variant'] = variant
    predictions['request_elapsed_seconds'] = elapsed
    return predictions


def run_local() -> dict:
    protocol, frame = load_run()
    config = protocol['config']
    output = RUN / 'local'
    output.mkdir(exist_ok=True)
    for variant in protocol['variants']:
        if variant['id'] not in ['calendar_30d', 'weather_30d'] and not variant['stress']:
            continue
        for name in ['ridge', 'lightgbm']:
            path = output / f"{name}-{variant['id']}.csv"
            if path.exists():
                continue
            records = []
            for target in config['targets']:
                model_path = SNAPSHOT / 'baselines' / f"{name}-{target}-{variant['mode']}.joblib"
                manifest = json.loads(model_path.with_suffix('.manifest.json').read_text())
                if digest(model_path) != manifest['artifact_sha256']:
                    raise ValueError('Baseline artifact hash mismatch')
                model = joblib.load(model_path)
                for origin in origins(protocol):
                    perturbed = stress_history(frame, origin, variant['stress'], protocol)
                    x = origin_features(perturbed, origin, target, config, variant['mode'])
                    started = time.perf_counter()
                    values = model.predict(x)
                    elapsed = time.perf_counter() - started
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite local predictions')
                    records.extend({'model': name, 'origin': origin.isoformat(), 'target': target,
                                    'timestamp_utc': timestamp.isoformat(), 'lead_hour': step + 1,
                                    'prediction': float(value), 'variant': variant['id'],
                                    'request_elapsed_seconds': elapsed} for step, (timestamp, value) in enumerate(zip(x.index, values)))
            pd.DataFrame(records).to_csv(path.with_suffix('.tmp'), index=False)
            path.with_suffix('.tmp').replace(path)
            print(f'Local completed: {path.name}', flush=True)
    for lag in [24, 168]:
        path = output / f'naive_{lag}.csv'
        if path.exists():
            continue
        records = []
        for origin in origins(protocol):
            index = pd.date_range(origin, periods=24, freq='h')
            for target in config['targets']:
                values = frame.reindex(index - pd.Timedelta(hours=lag))[target].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Incomplete seasonal naive inputs')
                records.extend({'model': f'naive_{lag}', 'origin': origin.isoformat(), 'target': target,
                                'timestamp_utc': timestamp.isoformat(), 'lead_hour': step+1,
                                'prediction': float(value), 'variant': 'calendar_30d',
                                'request_elapsed_seconds': None} for step, (timestamp, value) in enumerate(zip(index, values)))
        pd.DataFrame(records).to_csv(path, index=False)
    return {'status': 'completed', 'files': len(list(output.glob('*.csv')))}


def run_remote(limit: int = 990) -> dict:
    protocol, frame = load_run()
    config = protocol['config']
    load_dotenv()
    key = os.getenv('TFC_API_KEY')
    if not key:
        raise ValueError('Configure TFC_API_KEY in .env')
    output = RUN / 'remote'
    output.mkdir(exist_ok=True)
    completed = calls = 0
    status = {'status': 'running', 'completed': 0, 'expected': protocol['expected_api_requests']}
    with httpx.Client(timeout=180) as client:
        for variant in protocol['variants']:
            for origin in origins(protocol):
                identifier = f"{variant['id']}-{origin.date()}"
                cache = output / f'{identifier}.json'
                if cache.exists():
                    record = json.loads(cache.read_text())
                    parse_response(record['response'], origin, config['targets'], config)
                    completed += 1
                    continue
                if calls >= limit:
                    status.update(status='paused_request_limit', completed=completed)
                    write_json(RUN / 'status.json', status)
                    return status
                perturbed = stress_history(frame, origin, variant['stress'], protocol)
                payload = build_request(perturbed, origin, variant['days'], config, config['targets'], variant['mode'])
                request_bytes = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
                with gzip.open(output / f'{identifier}.request.json.gz', 'wb') as stream:
                    stream.write(request_bytes)
                started = time.perf_counter()
                try:
                    response = client.post(config['api_url'], params={'model': config['model']},
                                           headers={'Authorization': f'Bearer {key}'}, json=payload)
                    calls += 1
                    if response.status_code != 200:
                        raise RuntimeError(f'HTTP {response.status_code}')
                    body = response.json()
                    parse_response(body, origin, config['targets'], config)
                except (httpx.HTTPError, ValueError, KeyError, TypeError, RuntimeError) as error:
                    status.update(status='blocked_api', completed=completed, failed_case=identifier,
                                  error=str(error) if isinstance(error, RuntimeError) else type(error).__name__)
                    write_json(RUN / 'status.json', status)
                    print(json.dumps(status), flush=True)
                    return status
                elapsed = time.perf_counter() - started
                write_json(cache, {'response': body, 'variant': variant['id'], 'origin': origin.isoformat(),
                                   'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
                                   'elapsed_seconds': elapsed, 'received_at_utc': datetime.now(UTC).isoformat(),
                                   'hosted_model_alias': config['model']})
                completed += 1
                status.update(completed=completed, latest_case=identifier)
                write_json(RUN / 'status.json', status)
                if completed % 10 == 0:
                    print(f't0 completed {completed}/{status["expected"]}: {identifier}', flush=True)
    status.update(status='completed', completed=completed)
    write_json(RUN / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'local', 'remote', 'report'])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--limit', type=int, default=990)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be a positive number of new requests')
    if args.action == 'prepare':
        result = prepare_run()
    elif args.action == 'local':
        result = run_local()
    elif args.action == 'remote':
        if not args.live:
            parser.error('remote requires --live to send API requests')
        try:
            result = run_remote(args.limit)
        except KeyboardInterrupt:
            status_path = RUN / 'status.json'
            result = json.loads(status_path.read_text())
            result.update(status='paused_by_user', completed=len(list((RUN / 'remote').glob('*.json'))))
            write_json(status_path, result)
            print(json.dumps(result, indent=2))
            raise SystemExit(130) from None
    else:
        from src.benchmarks.report import build_report
        result = build_report()
    print(json.dumps(result, indent=2))
    if result.get('status') == 'blocked_api':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
