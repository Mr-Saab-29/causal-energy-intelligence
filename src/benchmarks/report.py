"""Compute E1–E8 from cached predictions and export a static dashboard snapshot."""
from __future__ import annotations

from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.run import RUN, load_run
from src.benchmarks.t0 import digest, load_frame, parse_response, write_json
from src.carbon.intensity import build_hourly_carbon_frame
from src.optimization.workload_shift import WorkloadConstraints, build_workload_decision_rankings, annotate_regret_and_savings

EXPERIMENTS = {
    'E1': 'Zero-shot comparison', 'E2': 'Covariate ablation', 'E3': 'Forecast horizon',
    'E4': 'History length', 'E5': 'Missing and noisy weather', 'E6': 'Distribution shift',
    'E7': 'Probabilistic quality', 'E8': 'Decision impact',
}


def forecast_metrics(frame: pd.DataFrame, scale: float | None) -> dict:
    error = frame.prediction.to_numpy() - frame.actual.to_numpy()
    result = {'rows': len(frame), 'days': int(frame.origin.nunique()),
              'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error**2).mean())),
              'mase': float(np.abs(error).mean() / scale) if scale else None,
              'coverage80': None, 'width80': None, 'pinball': None,
              'cdf10': None, 'cdf50': None, 'cdf90': None}
    quantiles = ['q0.1', 'q0.5', 'q0.9']
    if all(column in frame and frame[column].notna().all() for column in quantiles):
        losses = []
        for q in [.1, .5, .9]:
            residual = frame.actual.to_numpy() - frame[f'q{q}'].to_numpy()
            losses.append(np.maximum(q * residual, (q-1) * residual))
            result[f'cdf{int(q*100)}'] = float((residual <= 0).mean())
        result['pinball'] = float(np.mean(losses))
        result['coverage80'] = float(((frame.actual >= frame['q0.1']) & (frame.actual <= frame['q0.9'])).mean())
        result['width80'] = float((frame['q0.9'] - frame['q0.1']).mean())
    return result


def matched(frame: pd.DataFrame) -> pd.DataFrame:
    """Require equal origins across available model/configuration groups."""
    groups = [set(group.origin) for _, group in frame.groupby(['model', 'variant'])]
    common = set.intersection(*groups) if groups else set()
    return frame[frame.origin.isin(common)]


def read_predictions(protocol: dict, actual: pd.DataFrame):
    frames = [pd.read_csv(path) for path in sorted((RUN/'local').glob('*.csv'))]
    probability_root = RUN/'probabilistic_v1'
    if probability_root.exists():
        summary_path = probability_root/'summary.json'
        if not summary_path.exists():
            raise ValueError('Probabilistic run is incomplete; resume python -m src.benchmarks.probabilistic first')
        summary = json.loads(summary_path.read_text())
        if summary['status'] != 'completed' or len(summary['files']) != 16:
            raise ValueError('Incomplete probabilistic prediction set')
        if digest(probability_root/'protocol.json') != summary['protocol_sha256']:
            raise ValueError('Probabilistic protocol hash mismatch')
        probability_protocol = json.loads((probability_root/'protocol.json').read_text())
        if probability_protocol['parent_protocol_sha256'] != digest(RUN/'protocol.json'):
            raise ValueError('Probabilistic predictions belong to a different benchmark protocol')
        for item in summary['files']:
            path = probability_root/'local'/item['filename']
            if digest(path) != item['prediction_sha256']:
                raise ValueError('Probabilistic prediction hash mismatch')
            frames.append(pd.read_csv(path))
    timings = []
    for path in sorted((RUN/'remote').glob('*.json')):
        record = json.loads(path.read_text())
        request_path = path.with_name(path.stem + '.request.json.gz')
        with gzip.open(request_path, 'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest() != record['request_sha256']:
                raise ValueError('Cached API request hash mismatch')
        forecast = parse_response(record['response'], pd.Timestamp(record['origin']), protocol['config']['targets'], protocol['config'])
        forecast['variant'] = record['variant']
        forecast['request_elapsed_seconds'] = record['elapsed_seconds']
        frames.append(forecast)
        timings.append({'variant': record['variant'], 'origin': record['origin'], 'seconds': record['elapsed_seconds']})
    if not frames:
        raise ValueError('No predictions available')
    predictions = pd.concat(frames, ignore_index=True)
    predictions['timestamp_utc'] = pd.to_datetime(predictions.timestamp_utc, utc=True)
    predictions['origin'] = pd.to_datetime(predictions.origin, utc=True)
    keys = ['model', 'variant', 'origin', 'target', 'lead_hour']
    if predictions.duplicated(keys).any():
        raise ValueError('Duplicate predictions')
    counts = predictions.groupby(['model', 'variant', 'origin', 'target']).size()
    if not counts.eq(24).all():
        raise ValueError('Incomplete forecast blocks')
    quantile_columns = ['q0.1', 'q0.5', 'q0.9']
    if all(column in predictions for column in quantile_columns):
        has_quantiles = predictions[quantile_columns].notna().any(axis=1)
        values = predictions.loc[has_quantiles, quantile_columns].to_numpy()
        if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
            raise ValueError('Incomplete, crossed or nonfinite forecast quantiles')
    truth = actual.rename_axis('timestamp_utc').reset_index().melt(id_vars='timestamp_utc', var_name='target', value_name='actual')
    predictions = predictions.merge(truth, on=['timestamp_utc', 'target'], how='left', validate='many_to_one')
    if not np.isfinite(predictions[['actual', 'prediction']].to_numpy()).all():
        raise ValueError('Missing/nonfinite outcomes or predictions')
    return predictions, timings


def decision_days(predictions: pd.DataFrame, actual: pd.DataFrame, protocol: dict):
    price = load_frame(RUN/'prices.csv').price_eur_mwh
    factors = protocol['emission_factors']
    sources = list(factors)
    results, details = [], []
    reference = predictions[predictions.variant == 'calendar_30d']
    for (model, origin), block in reference.groupby(['model', 'origin']):
        index = pd.date_range(origin, periods=24, freq='h')
        expected = [f'{source}_mwh' for source in sources]
        wide = block.pivot(index='timestamp_utc', columns='target', values='prediction').reindex(index)[expected]
        truth = actual.reindex(index)[expected].copy()
        wide.columns = truth.columns = sources
        if not np.isfinite(wide.to_numpy()).all():
            continue
        carbon = build_hourly_carbon_frame(truth, wide, factors, 'direct_operational_emissions', origin.isoformat(), model)
        carbon['actual_price_eur_mwh'] = price.reindex(index).to_numpy()
        carbon['predicted_price_eur_mwh'] = price.reindex(index-pd.Timedelta(days=1)).to_numpy()
        carbon['previous_day_price_eur_mwh'] = carbon['predicted_price_eur_mwh']
        carbon['decision_date'] = origin.date().isoformat()
        if not np.isfinite(carbon[['actual_carbon_intensity_g_co2e_per_kwh', 'predicted_carbon_intensity_g_co2e_per_kwh']].to_numpy()).all():
            raise ValueError('Undefined carbon intensity: nonpositive total generation')
        ranked = build_workload_decision_rankings(carbon, WorkloadConstraints(price_weight=.2, carbon_weight=.8))
        # Explicit frozen research policy: no uncertainty penalty or learned overlay.
        ranked = ranked.sort_values('timestamp_utc').reset_index(drop=True)
        ranked['predicted_combined_score'] = ranked['base_predicted_combined_score']
        ranked['predicted_decision_rank'] = ranked.predicted_combined_score.rank(method='first').astype(int)
        annotate_regret_and_savings(ranked, ['window', 'model', 'decision_date'])
        chosen = ranked.loc[ranked.predicted_decision_rank.idxmin()]
        top = ranked[ranked.predicted_decision_rank <= 5]
        actual_top = set(ranked[ranked.actual_decision_rank <= 5].timestamp_utc)
        results.append({'model': model, 'origin': origin, 'variant': 'calendar_30d',
                        'combined_regret': float(chosen.combined_regret),
                        'carbon_regret': float(chosen.carbon_regret_g_co2e_per_kwh),
                        'cost_regret': float(chosen.cost_regret_eur_mwh),
                        'carbon_savings': float(chosen.carbon_savings_vs_run_now_g_co2e_per_kwh),
                        'cost_savings': float(chosen.cost_savings_vs_run_now_eur_mwh),
                        'top5_overlap': len(set(top.timestamp_utc) & actual_top) / 5,
                        'best_hour_capture': float(ranked.loc[ranked.actual_decision_rank.idxmin(), 'predicted_decision_rank'] <= 5)})
        details.append({'model': model, 'origin': origin.isoformat(), 'chosen_hour': chosen.timestamp_utc.isoformat(),
                        'oracle_hour': ranked.loc[ranked.actual_decision_rank.idxmin(), 'timestamp_utc'].isoformat(),
                        'top5_hours': [x.isoformat() for x in top.sort_values('predicted_decision_rank').timestamp_utc],
                        'selected_carbon': float(chosen.actual_avg_carbon_intensity_g_co2e_per_kwh),
                        'run_now_carbon': float(chosen.run_now_carbon_intensity_g_co2e_per_kwh),
                        'selected_cost': float(chosen.actual_avg_price_eur_mwh),
                        'run_now_cost': float(chosen.run_now_price_eur_mwh)})
    return pd.DataFrame(results), details


def summarize(predictions: pd.DataFrame, actual: pd.DataFrame, protocol: dict):
    rows = []
    end = pd.Timestamp(protocol['config']['evaluation_start']) + pd.Timedelta(days=90)
    regimes = {'all': pd.Series(True, index=actual.index)}
    for column, labels in [('consumption_mwh', ('low_demand', 'high_demand')),
                           ('avg_temperature_c', ('cold', 'hot')),
                           ('total_production_mwh', ('low_generation', 'high_generation')),
                           ('wind_mwh', ('low_wind', 'high_wind'))]:
        threshold = protocol['regime_thresholds'][column]
        regimes[labels[0]] = actual[column] < threshold['p05']
        regimes[labels[1]] = actual[column] > threshold['p95']
    def add(exp, frame, window, horizon=24, regime='all'):
        if frame.empty:
            return
        for (model, variant, target), group in frame.groupby(['model', 'variant', 'target']):
            row = {'experiment': exp, 'window_days': window, 'model': model, 'variant': variant,
                   'target': target, 'horizon': horizon, 'regime': regime,
                   **forecast_metrics(group, protocol['mase_scales'][target])}
            row['evidence'] = 'limited' if row['days'] < 10 else 'available'
            rows.append(row)
    for window in [90, 28, 7]:
        selected = predictions[predictions.origin >= end-pd.Timedelta(days=window)]
        reference = matched(selected[selected.variant == 'calendar_30d'])
        add('E1', reference, window)
        add('E2', matched(selected[selected.variant.isin(['target_30d', 'calendar_30d', 'weather_30d', 'related_30d'])]), window)
        for horizon in [6, 12, 24]:
            add('E3', reference[reference.lead_hour <= horizon], window, horizon)
        add('E4', matched(selected[(selected.model == 't0-alpha') & selected.variant.isin(['calendar_7d', 'calendar_30d', 'calendar_90d'])]), window)
        add('E5', matched(selected[selected.variant.isin(['weather_30d', 'dropout_10', 'dropout_30', 'outage_6h', 'noise_10', 'noise_30'])]), window)
        for regime, mask in regimes.items():
            add('E6', reference[reference.timestamp_utc.isin(actual.index[mask])], window, regime=regime)
        for horizon in [6, 12, 24]:
            for regime, mask in regimes.items():
                quantile_columns = ['q0.1', 'q0.5', 'q0.9']
                available = selected.reindex(columns=quantile_columns).notna().all(axis=1)
                probabilistic = matched(selected[available & (selected.variant == 'calendar_30d') & (selected.lead_hour <= horizon)])
                add('E7', probabilistic[probabilistic.timestamp_utc.isin(actual.index[mask])], window, horizon, regime)
    return rows


def build_report():
    protocol, actual = load_run()
    predictions, timings = read_predictions(protocol, actual)
    metrics = summarize(predictions, actual, protocol)
    decisions, details = decision_days(predictions, actual, protocol)
    end = pd.Timestamp(protocol['config']['evaluation_start']) + pd.Timedelta(days=90)
    if not decisions.empty:
        for window in [90, 28, 7]:
            selected = matched(decisions[decisions.origin >= end-pd.Timedelta(days=window)])
            for model, group in selected.groupby('model'):
                metrics.append({'experiment': 'E8', 'window_days': window, 'model': model, 'variant': 'calendar_30d',
                                'target': 'all_sources', 'horizon': 24, 'regime': 'all', 'days': len(group),
                                **{name: float(group[name].mean()) for name in ['combined_regret', 'carbon_regret', 'cost_regret',
                                    'carbon_savings', 'cost_savings', 'top5_overlap', 'best_hour_capture']}})
    status = json.loads((RUN/'status.json').read_text())
    coverage = predictions.groupby(['model', 'variant']).origin.nunique()
    availability = [{'model': 't0-alpha', 'variant': variant['id'],
                     'completed_days': int(coverage.get(('t0-alpha', variant['id']), 0)), 'expected_days': 90}
                    for variant in protocol['variants']]
    local_complete = len(list((RUN/'local').glob('*.csv'))) == 16
    remote_complete = all(item['completed_days'] == item['expected_days'] for item in availability)
    timing_summary = []
    if timings:
        for variant, group in pd.DataFrame(timings).groupby('variant'):
            timing_summary.append({'model': 't0-alpha', 'variant': variant, 'unit': 'API request / 10 targets',
                                   'calls': len(group), 'median_seconds': float(group.seconds.median()),
                                   'p95_seconds': float(group.seconds.quantile(.95))})
    local_times = predictions[(~predictions.model.isin(['t0-alpha', 'naive_24', 'naive_168'])) & (predictions.variant == 'calendar_30d')].drop_duplicates(['model', 'origin', 'target'])
    for model, group in local_times.groupby('model'):
        timing_summary.append({'model': model, 'variant': 'calendar_30d', 'unit': 'Local predict / 1 target',
                               'calls': len(group), 'median_seconds': float(group.request_elapsed_seconds.median()),
                               'p95_seconds': float(group.request_elapsed_seconds.quantile(.95))})
    payload = {'schema_version': 1, 'generated_at_utc': datetime.now(UTC).isoformat(),
               'status': 'completed' if remote_complete and local_complete else 'partial',
               'run_status': status, 'date_start': protocol['config']['evaluation_start'],
               'date_end': (end-pd.Timedelta(hours=1)).isoformat(), 'snapshot_id': RUN.parent.name,
               'models': sorted(predictions.model.unique()), 'targets': protocol['config']['targets'],
               'experiments': EXPERIMENTS, 'availability': availability, 'metrics': metrics,
               'decision_examples': details, 'timings': timings, 'timing_summary': timing_summary, 'protocol': protocol,
               'probabilistic_protocol': json.loads((RUN/'probabilistic_v1/protocol.json').read_text())
                   if (RUN/'probabilistic_v1/summary.json').exists() else None,
               'limitations': [
                   'Frozen retrospective benchmark; source publication vintages and pretraining overlap are unverified.',
                   't0-alpha is a hosted alias; predictions are cached but checkpoint revision is not pinned.',
                   'LightGBM and Ridge are isolated retrained benchmark variants, not the original production checkpoints.',
                   'E2 untrained baseline input configurations and E4 baseline context ablations are N/A.',
                   'E5 uses historical weather corruption followed by causal imputation, not native missing-value support.',
                   'E7 compares t0 and available probabilistic local variants on the same quantile levels and matched origins; original point models remain N/A.',
                   'Probabilistic LightGBM uses native quantile objectives. Ridge and seasonal-naive variants use lead-specific held-out residual distributions.',
                   'Ridge residual variants fit before the final 28 pre-evaluation days reserved for calibration; original Ridge fits the entire pre-period.',
                   'Probabilistic variants use q0.5 for point metrics and E8, so their MAE/RMSE and decisions may differ from original point models.',
                   'Nominal 80% coverage is a target to evaluate, not a guarantee under temporal dependence or distribution shift.',
                   'Rows use common origins across available comparison groups; incomplete configurations remain marked pending.',
                   'E8 reuses carbon accounting and ordinal ranking without learned overlays or uncertainty penalties.',
                   'API timing is end-to-end for a ten-series request; local timing is per-target prediction only and is not directly comparable.',
                   'Intervals for total carbon are not inferred by adding marginal source quantiles.',
               ]}
    predictions.to_csv(RUN/'predictions.csv.gz', index=False, compression='gzip')
    pd.DataFrame(metrics).to_csv(RUN/'metrics.csv', index=False)
    if not decisions.empty:
        decisions.to_csv(RUN/'decision_days.csv', index=False)
    write_json(RUN/'benchmark.json', payload)
    destination = Path('frontend/public/data/benchmark.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, payload)
    return {'status': payload['status'], 'prediction_rows': len(predictions), 'metric_rows': len(metrics),
            'remote_requests': len(timings), 'dashboard_artifact': str(destination)}
