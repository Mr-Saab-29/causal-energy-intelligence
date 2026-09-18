"""Tests for repaired observations, frozen-origin features and diagnostic controls."""
import numpy as np
import pandas as pd
import pytest

from src.benchmarks.baselines import origin_features, training_examples
from src.benchmarks.recover import fill_observed
from src.benchmarks.verify_covariates import compare_predictions, diagnostic_requests


@pytest.fixture
def data():
    index = pd.date_range('2026-01-01', periods=24 * 50, freq='h', tz='UTC')
    frame = pd.DataFrame({'load': np.arange(len(index), dtype=float),
                          'wind': np.arange(len(index), dtype=float) * 2,
                          'weather': np.sin(np.arange(len(index)))}, index=index)
    config = {'targets': ['load', 'wind'], 'weather': ['weather'], 'horizon_hours': 24,
              'history_forward_fill_limit_hours': 24, 'quantiles': [0.1, 0.5, 0.9],
              'evaluation_start': '2026-02-10T00:00:00Z'}
    return frame, config


def test_recovery_never_overwrites_observed_values():
    original = pd.DataFrame({'x': [1., np.nan, np.nan]}, index=[1, 2, 3])
    source = pd.DataFrame({'x': [100., 2.]}, index=[1, 2])
    result, count = fill_observed(original, source)
    assert count == 1
    assert result.x.iloc[:2].tolist() == [1., 2.]
    assert pd.isna(result.x.iloc[2])
    with pytest.raises(ValueError, match='Duplicate'):
        fill_observed(original, pd.concat([source, source]))


def test_all_baseline_leads_obey_origin_cutoff(data):
    frame, config = data
    origin = pd.Timestamp(config['evaluation_start'])
    expected = origin_features(frame, origin, 'load', config, 'historical_weather')
    frame.loc[origin:] = -999999
    actual = origin_features(frame, origin, 'load', config, 'historical_weather')
    pd.testing.assert_frame_equal(expected, actual)
    assert actual.lag_24h.iloc[-1] == frame.loc[origin - pd.Timedelta(hours=1), 'load']
    assert actual.lag_168h.iloc[0] == frame.loc[origin - pd.Timedelta(days=7), 'load']


def test_training_never_reads_evaluation_values(data):
    frame, config = data
    x, y, origins = training_examples(frame, 'load', config, 'calendar')
    cutoff = pd.Timestamp(config['evaluation_start'])
    assert x.index.max() < cutoff
    assert y.index.max() < cutoff
    frame.loc[cutoff:] = np.nan
    x2, y2, origins2 = training_examples(frame, 'load', config, 'calendar')
    pd.testing.assert_frame_equal(x, x2)
    pd.testing.assert_series_equal(y, y2)
    assert origins == origins2


def test_training_labels_are_never_imputed(data):
    frame, config = data
    frame.loc['2026-01-20 05:00:00+00:00', 'load'] = np.nan
    _, y, origins = training_examples(frame, 'load', config, 'calendar')
    assert '2026-01-20T00:00:00+00:00' not in origins
    assert y.notna().all()


def test_diagnostic_changes_covariates_not_target_or_origin(data):
    frame, config = data
    origin = pd.Timestamp(config['evaluation_start']) - pd.Timedelta(days=1)
    requests = diagnostic_requests(frame, origin, config)
    reference = requests['target_only']['series'][0]
    assert requests['target_only'] == requests['target_only_repeat']
    for request in requests.values():
        item = request['series'][0]
        assert item['target'] == reference['target']
        assert item['index'] == reference['index']
        assert max(pd.to_datetime(item['index'], utc=True)) < origin
    related = requests['related_history']['series'][0]['hist_variables']
    assert set(related) == {'wind'}
    before = requests
    frame.loc[origin:] = 9999
    assert before == diagnostic_requests(frame, origin, config)


def test_diagnostic_does_not_overclaim_nondeterministic_changes():
    names = ['target_only', 'target_only_repeat', 'calendar', 'historical_weather',
             'weather_shuffled', 'related_history', 'related_shuffled']
    predictions = {name: np.ones(24) for name in names}
    predictions['target_only_repeat'] += 1
    predictions['calendar'] += 2
    report = compare_predictions(predictions)
    assert not report['comparisons']['calendar']['observed_sensitivity_above_repeat_tolerance']


def test_related_adapter_excludes_self_and_future(data):
    from src.benchmarks.t0 import build_request
    frame, config = data
    origin = pd.Timestamp(config['evaluation_start'])
    payload = build_request(frame, origin, 30, config, ['load', 'wind'], 'related_history')
    assert set(payload['series'][0]['hist_variables']) == {'wind'}
    assert set(payload['series'][1]['hist_variables']) == {'load'}
    frame.loc[origin:] = 999999
    assert payload == build_request(frame, origin, 30, config, ['load', 'wind'], 'related_history')
