"""Train and forecast probabilistic local benchmark variants; never call the t0 API.

python -m src.benchmarks.probabilistic
Artifacts and predictions are isolated from the original point-model benchmark.
"""
from __future__ import annotations

from datetime import UTC, datetime
import json
from importlib.metadata import version
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmarks.baselines import origin_features
from src.benchmarks.recover import SNAPSHOT
from src.benchmarks.run import RUN, load_run, origins, stress_history
from src.benchmarks.t0 import digest, write_json

QUANTILES = [.1, .5, .9]
CALIBRATION_DAYS = 28
MIN_CALIBRATION_ORIGINS = 20
MODELS = ['lightgbm_quantile', 'ridge_residual', 'naive_24_residual', 'naive_168_residual']
OUTPUT = RUN / 'probabilistic_v1'


def read_training(target: str, mode: str, cutoff: pd.Timestamp):
    path = SNAPSHOT / 'baselines' / f'{target}-{mode}.training.csv.gz'
    manifest = json.loads((SNAPSHOT / 'baselines' / f'ridge-{target}-{mode}.manifest.json').read_text())
    if digest(path) != manifest['training_data_sha256']:
        raise ValueError('Baseline training table hash mismatch')
    table = pd.read_csv(path)
    timestamps = pd.to_datetime(table['timestamp_utc'], utc=True)
    forecast_origins = pd.to_datetime(table['origin'], utc=True)
    if not (timestamps < cutoff).all() or not (forecast_origins < cutoff).all():
        raise ValueError('Training/calibration data overlaps evaluation')
    if not forecast_origins.groupby(forecast_origins).size().eq(24).all():
        raise ValueError('Calibration requires complete 24-hour origin blocks')
    expected_lead = ((timestamps - forecast_origins) / pd.Timedelta(hours=1)).astype(int) + 1
    if not expected_lead.equals(table['lead_hour'].astype(int)):
        raise ValueError('Training forecast leads are not aligned with origins')
    x = table[manifest['features']]
    y = table['target_value']
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite pre-period training inputs')
    return x, y, forecast_origins, timestamps, digest(path)


def residual_offsets(residual: np.ndarray, leads: pd.Series) -> tuple[np.ndarray, list[int]]:
    """Empirical residual quantiles by lead; no claim of conformal coverage."""
    offsets, counts = [], []
    for lead in range(1, 25):
        values = np.asarray(residual)[np.asarray(leads) == lead]
        if len(values) < MIN_CALIBRATION_ORIGINS or not np.isfinite(values).all():
            raise ValueError(f'Lead {lead} needs at least {MIN_CALIBRATION_ORIGINS} finite calibration days')
        offsets.append(np.quantile(values, QUANTILES).tolist())
        counts.append(len(values))
    return np.asarray(offsets), counts


def predict_quantiles(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    model = bundle['model']
    if model == 'lightgbm_quantile':
        values = np.column_stack([estimator.predict(x) for estimator in bundle['estimators']])
        # Independent quantile regressors can cross. Apply fixed monotone rearrangement.
        values = np.sort(values, axis=1)
    else:
        if model == 'ridge_residual':
            center = bundle['estimator'].predict(x)
        else:
            lag = 24 if model == 'naive_24_residual' else 168
            center = x[f'lag_{lag}h'].to_numpy()
        lead_index = x['lead_hour'].to_numpy(dtype=int) - 1
        if ((lead_index < 0) | (lead_index >= 24)).any():
            raise ValueError('Unsupported forecast lead')
        values = np.asarray(center)[:, None] + bundle['offsets'][lead_index]
    if values.shape != (len(x), 3) or not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError('Invalid probabilistic predictions')
    return values


def fit_bundle(model: str, target: str, mode: str, config: dict) -> tuple[dict, dict]:
    cutoff = pd.Timestamp(config['evaluation_start'])
    calibration_start = cutoff - pd.Timedelta(days=CALIBRATION_DAYS)
    x, y, forecast_origins, timestamps, training_hash = read_training(target, mode, cutoff)
    calibration = forecast_origins >= calibration_start
    fitting = timestamps < calibration_start
    if ((fitting & calibration).any() or not fitting.any() or not calibration.any()):
        raise ValueError('Invalid chronological fit/calibration split')
    metadata = {'model': model, 'target': target, 'mode': mode, 'quantiles': QUANTILES,
                'evaluation_start': cutoff.isoformat(), 'training_data_sha256': training_hash,
                'feature_columns': list(x.columns), 'point_forecast': 'q0.5',
                'code_sha256': digest(Path(__file__)), 'fitted_at_utc': datetime.now(UTC).isoformat(),
                'selection': 'fixed methods and parameters; evaluation never used for fitting or calibration'}
    bundle = {'model': model}
    if model == 'lightgbm_quantile':
        estimators = []
        for q in QUANTILES:
            estimator = LGBMRegressor(objective='quantile', alpha=q, n_estimators=600,
                                     learning_rate=.03, num_leaves=31, subsample=.9,
                                     colsample_bytree=.9, reg_alpha=.1, reg_lambda=1.,
                                     random_state=42, n_jobs=2, verbosity=-1)
            estimator.fit(x, y)
            estimators.append(estimator)
        bundle['estimators'] = estimators
        metadata.update(method='native LightGBM quantile regression; row-wise sorting prevents crossing',
                        fit_label_end=timestamps.max().isoformat(), fit_rows=len(x),
                        calibration_start=None, calibration_end=None,
                        parameters=[estimator.get_params() for estimator in estimators])
    else:
        if model == 'ridge_residual':
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            estimator.fit(x.loc[fitting], y.loc[fitting])
            center = estimator.predict(x.loc[calibration])
            bundle['estimator'] = estimator
            metadata.update(fit_label_end=timestamps[fitting].max().isoformat(), fit_rows=int(fitting.sum()),
                            parameters={'alpha': 1., 'standardization': 'fitting subset only'})
        elif model in {'naive_24_residual', 'naive_168_residual'}:
            lag = 24 if model == 'naive_24_residual' else 168
            center = x.loc[calibration, f'lag_{lag}h'].to_numpy()
            metadata.update(fit_label_end=None, fit_rows=0, parameters={'season_hours': lag})
        else:
            raise ValueError('Unknown local probabilistic model')
        offsets, counts = residual_offsets(y.loc[calibration].to_numpy() - center, x.loc[calibration, 'lead_hour'])
        bundle['offsets'] = offsets
        metadata.update(method='add lead-specific empirical held-out residual quantiles; no guaranteed coverage',
                        calibration_start=calibration_start.isoformat(), calibration_end=timestamps[calibration].max().isoformat(),
                        calibration_counts_by_lead=counts, residual_offsets=offsets.tolist(),
                        refit_after_calibration=False)
    return bundle, metadata


def load_or_train(model: str, target: str, mode: str, config: dict) -> dict:
    path = OUTPUT / 'models' / f'{model}-{target}-{mode}.joblib'
    manifest_path = path.with_suffix('.manifest.json')
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if digest(path) != manifest['artifact_sha256'] or manifest['code_sha256'] != digest(Path(__file__)):
            raise ValueError('Probabilistic artifact/code changed; use a new output version')
        training_path = SNAPSHOT / 'baselines' / f'{target}-{mode}.training.csv.gz'
        if digest(training_path) != manifest['training_data_sha256']:
            raise ValueError('Probabilistic training source changed')
        return joblib.load(path)
    bundle, manifest = fit_bundle(model, target, mode, config)
    temporary = path.with_suffix('.tmp')
    joblib.dump(bundle, temporary)
    temporary.replace(path)
    manifest['artifact_sha256'] = digest(path)
    write_json(manifest_path, manifest)
    print(f'Fitted {model} / {target} / {mode}', flush=True)
    return bundle


def run() -> dict:
    protocol, frame = load_run()
    config = protocol['config']
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'models').mkdir(exist_ok=True)
    local = OUTPUT/'local'
    local.mkdir(exist_ok=True)
    run_protocol = {'version': 1, 'quantiles': QUANTILES, 'models': MODELS,
                    'calibration_days': CALIBRATION_DAYS, 'minimum_calibration_origins': MIN_CALIBRATION_ORIGINS,
                    'parent_protocol_sha256': digest(RUN/'protocol.json'), 'implementation_sha256': digest(Path(__file__)),
                    'feature_implementation_sha256': digest(Path('src/benchmarks/baselines.py')),
                    'stress_implementation_sha256': digest(Path('src/benchmarks/run.py')),
                    'packages': {name: version(name) for name in ['lightgbm', 'scikit-learn', 'numpy', 'pandas']},
                    'point_forecast': 'median; new model IDs preserve old point-model comparisons',
                    'lightgbm_fit': 'all pre-evaluation labels; native quantile objectives',
                    'ridge_fit': 'labels before held-out calibration period; never refit after calibration',
                    'residual_calibration': 'final 28 pre-evaluation days, separate empirical distribution at each lead',
                    'guaranteed_coverage': False}
    protocol_path = OUTPUT/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != run_protocol:
        raise ValueError('Probabilistic protocol/code changed; use a new output version')
    write_json(protocol_path, run_protocol)
    completed = []
    for model in MODELS:
        for variant in protocol['variants']:
            if variant['id'] not in ['calendar_30d', 'weather_30d'] and not variant['stress']:
                continue
            if model.startswith('naive_') and variant['id'] != 'calendar_30d':
                continue
            path = local/f"{model}-{variant['id']}.csv"
            metadata_path = path.with_suffix('.metadata.json')
            if path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if digest(path) != metadata['prediction_sha256']:
                    raise ValueError('Probabilistic predictions changed')
                completed.append(metadata)
                continue
            records = []
            for target in config['targets']:
                bundle = load_or_train(model, target, variant['mode'], config)
                for origin in origins(protocol):
                    inputs = stress_history(frame, origin, variant['stress'], protocol)
                    x = origin_features(inputs, origin, target, config, variant['mode'])
                    started = time.perf_counter()
                    values = predict_quantiles(bundle, x)
                    elapsed = time.perf_counter() - started
                    for lead, (timestamp, quantiles) in enumerate(zip(x.index, values), start=1):
                        records.append({'model': model, 'variant': variant['id'], 'target': target,
                                        'origin': origin.isoformat(), 'timestamp_utc': timestamp.isoformat(),
                                        'lead_hour': lead, 'prediction': float(quantiles[1]),
                                        **{f'q{q}': float(v) for q, v in zip(QUANTILES, quantiles)},
                                        'request_elapsed_seconds': elapsed})
            predictions = pd.DataFrame(records)
            temporary = path.with_suffix('.tmp')
            predictions.to_csv(temporary, index=False)
            temporary.replace(path)
            metadata = {'filename': path.name, 'prediction_sha256': digest(path), 'rows': len(predictions),
                        'model': model, 'variant': variant['id']}
            write_json(metadata_path, metadata)
            completed.append(metadata)
            print(f'Saved probabilistic forecasts: {path.name}', flush=True)
    summary = {'status': 'completed', 'files': completed, 'protocol_sha256': digest(protocol_path),
               'created_at_utc': datetime.now(UTC).isoformat(), 'api_requests': 0}
    write_json(OUTPUT/'summary.json', summary)
    return {'status': 'completed', 'prediction_files': len(completed), 'output': str(OUTPUT), 'api_requests': 0}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
