"""Fixed direct-horizon baselines trained only before the frozen evaluation period."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmarks.recover import SNAPSHOT
from src.benchmarks.rebuild_energy import verify_energy_source
from src.benchmarks.t0 import digest, load_frame, slices, write_json


def origin_features(frame: pd.DataFrame, origin: pd.Timestamp, target: str,
                    config: dict, mode: str = 'calendar') -> pd.DataFrame:
    """Construct all 24 lead features from one fixed 7-day information cutoff."""
    if config['horizon_hours'] != 24 or mode not in {'calendar', 'historical_weather'}:
        raise ValueError('Baselines require a 24h horizon and supported mode')
    _, history, _ = slices(frame, origin, 7, config)
    y = history[target]
    if not np.isfinite(y).all():
        raise ValueError('Incomplete target history')
    index = pd.date_range(origin, periods=24, freq='h')
    local = index.tz_convert('Europe/Paris')
    data = {'lead_hour': np.arange(1, 25),
            'hour_sin': np.sin(2 * np.pi * local.hour / 24),
            'hour_cos': np.cos(2 * np.pi * local.hour / 24),
            'day_of_week': local.dayofweek,
            'month': local.month,
            'lag_24h': y.iloc[-24:].to_numpy(),
            'lag_168h': y.iloc[:24].to_numpy(),
            'last_observation': float(y.iloc[-1]),
            'history_mean_24h': float(y.iloc[-24:].mean()),
            'history_std_24h': float(y.iloc[-24:].std()),
            'history_mean_168h': float(y.mean())}
    if mode == 'historical_weather':
        weather = history[config['weather']]
        if not np.isfinite(weather.to_numpy()).all():
            raise ValueError('Incomplete historical weather')
        for column in config['weather']:
            data[f'{column}_last'] = float(weather[column].iloc[-1])
            data[f'{column}_mean_24h'] = float(weather[column].iloc[-24:].mean())
            data[f'{column}_lag_24h'] = weather[column].iloc[-24:].to_numpy()
    return pd.DataFrame(data, index=index)


def training_examples(frame: pd.DataFrame, target: str, config: dict, mode: str):
    cutoff = pd.Timestamp(config['evaluation_start'])
    # Slice first: even preprocessing cannot inspect evaluation observations.
    pre = frame.loc[frame.index < cutoff]
    xs, ys, origins = [], [], []
    start = pre.index.min().ceil('D') + pd.Timedelta(days=7)
    for origin in pd.date_range(start, cutoff - pd.Timedelta(days=1), freq='D'):
        try:
            features = origin_features(pre, origin, target, config, mode)
        except ValueError:
            continue
        labels = pre.reindex(features.index)[target]
        if not np.isfinite(labels).all():
            continue
        xs.append(features)
        ys.append(labels)
        origins.extend([origin.isoformat()] * 24)
    if not xs:
        raise ValueError('No eligible pre-period training examples')
    return pd.concat(xs), pd.concat(ys), origins


def train(snapshot: Path = SNAPSHOT) -> dict:
    config = json.loads((snapshot / 'config.json').read_text())
    verify_energy_source(config)
    audit = json.loads((snapshot / 'audit.json').read_text())
    source = Path(config['source'])
    if digest(source) != audit['source_sha256']:
        raise ValueError('Training source changed since freeze')
    output = snapshot / 'baselines'
    if output.exists():
        raise ValueError('Baseline artifacts already exist; preserve completed training')
    output.mkdir()
    frame = load_frame(source)
    cutoff = pd.Timestamp(config['evaluation_start'])
    before = frame.loc[frame.index < cutoff]
    before.rename_axis('timestamp_utc').to_csv(output / 'training_source.csv')
    production = {str(p): digest(p) for p in Path('models').glob('*.joblib')}
    records = []
    # Hyperparameters fixed in advance. No test scores or test-selected winner.
    for target in config['targets']:
        for mode in ['calendar', 'historical_weather']:
            x, y, origins = training_examples(before, target, config, mode)
            data = x.copy()
            data['target_value'] = y
            data['origin'] = origins
            training_path = output / f'{target}-{mode}.training.csv.gz'
            data.rename_axis('timestamp_utc').to_csv(training_path, compression='gzip')
            algorithms = {
                'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                'lightgbm': LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                         subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                                         reg_lambda=1.0, random_state=42, n_jobs=2, verbosity=-1),
            }
            for name, model in algorithms.items():
                model.fit(x, y)
                path = output / f'{name}-{target}-{mode}.joblib'
                joblib.dump(model, path)
                manifest = {'algorithm': name, 'target': target, 'mode': mode,
                            'created_at_utc': datetime.now(UTC).isoformat(),
                            'training_cutoff_exclusive': cutoff.isoformat(),
                            'label_start': x.index.min().isoformat(), 'label_end': x.index.max().isoformat(),
                            'training_rows': len(x), 'training_origins': len(set(origins)),
                            'features': list(x.columns), 'context_days': 7, 'horizon_hours': 24,
                            'selection': 'fixed algorithms and parameters; no tuning or winner selection',
                            'training_data_sha256': digest(training_path), 'artifact_sha256': digest(path),
                            'training_source_sha256': digest(output / 'training_source.csv'),
                            'benchmark_source_sha256': audit['source_sha256'],
                            'implementation_sha256': digest(Path(__file__)),
                            'parameters': {k: v for k, v in model.get_params().items()
                                           if isinstance(v, (str, int, float, bool, type(None)))}}
                write_json(path.with_suffix('.manifest.json'), manifest)
                records.append(manifest)
                print(f'Trained {name} / {target} / {mode}: {len(x)} rows', flush=True)
    # Smoke-test inference on a full origin with no evaluation-based selection.
    eligible = pd.read_csv(snapshot / 'origins.csv')
    origin = pd.Timestamp(eligible[(eligible.context_days == 7) & eligible.eligible & eligible.weather_history_eligible].iloc[0].origin)
    rows = []
    for manifest in records:
        target, mode, name = manifest['target'], manifest['mode'], manifest['algorithm']
        path = output / f'{name}-{target}-{mode}.joblib'
        model = joblib.load(path)
        x = origin_features(frame, origin, target, config, mode)
        predictions = model.predict(x)
        if not np.isfinite(predictions).all():
            raise ValueError('Nonfinite baseline smoke predictions')
        for timestamp, value in zip(x.index, predictions):
            rows.append({'origin': origin.isoformat(), 'timestamp_utc': timestamp.isoformat(),
                         'model': name, 'mode': mode, 'target': target, 'prediction': float(value)})
    pd.DataFrame(rows).to_csv(output / 'smoke_predictions.csv', index=False)
    summary = {'status': 'completed', 'artifacts': len(records), 'smoke_rows': len(rows),
               'training_cutoff_exclusive': cutoff.isoformat(), 'evaluation_used_for_selection': False,
               'production_artifacts_unchanged': all(digest(Path(p)) == h for p, h in production.items()),
               'baseline_label': 'benchmark-specific retrained LightGBM and Ridge, not original production checkpoints'}
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(train(), indent=2))
