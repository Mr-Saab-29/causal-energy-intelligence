import numpy as np
import pandas as pd
import pytest

from src.benchmarks.run import stress_history


@pytest.mark.parametrize('stress', ['dropout_10', 'dropout_30', 'outage_6h', 'noise_10', 'noise_30'])
def test_stress_only_changes_past_weather(stress):
    index = pd.date_range('2026-01-01', periods=40*24, freq='h', tz='UTC')
    frame = pd.DataFrame({'target': 5., 'weather': np.sin(np.arange(len(index)))}, index=index)
    origin = index[35*24]
    protocol = {'seed': 42, 'config': {'weather': ['weather']},
                'weather_statistics': {'weather': {'mean': 0., 'std': 1.}}}
    result = stress_history(frame, origin, stress, protocol)
    pd.testing.assert_series_equal(result.target, frame.target)
    pd.testing.assert_frame_equal(result.loc[origin:], frame.loc[origin:])
    pd.testing.assert_frame_equal(result, stress_history(frame, origin, stress, protocol))
    assert np.isfinite(result.to_numpy()).all()
    assert not result.weather.equals(frame.weather)


def test_metrics_quantiles_and_zero_scale():
    from src.benchmarks.report import forecast_metrics
    frame = pd.DataFrame({'origin': ['a', 'a'], 'prediction': [1., 3.], 'actual': [2., 2.],
                          'q0.1': [0., 1.], 'q0.5': [1., 3.], 'q0.9': [2., 4.]})
    metrics = forecast_metrics(frame, 2.)
    assert metrics['mae'] == 1
    assert metrics['rmse'] == 1
    assert metrics['mase'] == .5
    assert metrics['coverage80'] == 1
    assert metrics['width80'] == 2.5
    assert metrics['cdf50'] == .5
    assert forecast_metrics(frame, None)['mase'] is None
    point = forecast_metrics(frame.drop(columns=['q0.1']), 2.)
    assert point['coverage80'] is None


def test_matching_uses_common_days():
    from src.benchmarks.report import matched
    frame = pd.DataFrame({'model': ['a', 'a', 'b'], 'variant': ['x'] * 3, 'origin': ['day1', 'day2', 'day2']})
    result = matched(frame)
    assert len(result) == 2
    assert set(result.origin) == {'day2'}


def test_perfect_generation_selects_oracle_with_shared_prices(tmp_path, monkeypatch):
    import src.benchmarks.report as module
    index = pd.date_range('2026-05-26', periods=24, freq='h', tz='UTC')
    sources = ['nuclear', 'gas', 'coal', 'oil', 'wind', 'solar', 'hydro', 'bioenergy']
    factors = {name: 0. for name in sources}
    factors['gas'] = 370.
    actual = pd.DataFrame({f'{source}_mwh': 100. for source in sources}, index=index)
    actual['gas_mwh'] = np.arange(24) * 20.
    predictions = []
    for target in actual:
        for hour, timestamp in enumerate(index):
            predictions.append({'model': 'perfect', 'variant': 'calendar_30d', 'origin': index[0],
                                'target': target, 'timestamp_utc': timestamp, 'prediction': actual.loc[timestamp, target]})
    price_index = pd.date_range(index[0]-pd.Timedelta(days=1), periods=48, freq='h')
    pd.DataFrame({'timestamp_utc': price_index, 'price_eur_mwh': 50.}).to_csv(tmp_path/'prices.csv', index=False)
    monkeypatch.setattr(module, 'RUN', tmp_path)
    metrics, details = module.decision_days(pd.DataFrame(predictions), actual, {'emission_factors': factors})
    assert metrics.iloc[0].combined_regret == 0
    assert metrics.iloc[0].carbon_regret == 0
    assert metrics.iloc[0].top5_overlap == 1
    assert details[0]['chosen_hour'] == details[0]['oracle_hour']


def test_api_runner_stops_on_quota_error(tmp_path, monkeypatch):
    import httpx
    import src.benchmarks.run as module
    index = pd.date_range('2026-01-01', periods=32*24, freq='h', tz='UTC')
    frame = pd.DataFrame({'load': 1.}, index=index)
    config = {'evaluation_start': index[30*24].isoformat(), 'evaluation_days': 2,
              'targets': ['load'], 'weather': [], 'horizon_hours': 24,
              'history_forward_fill_limit_hours': 24, 'quantiles': [.1, .5, .9],
              'model': 't0-alpha', 'api_url': 'https://api.retrocast.com/forecast'}
    protocol = {'config': config, 'expected_api_requests': 2,
                'variants': [{'id': 'target_30d', 'mode': 'target_only', 'days': 30, 'stress': None}]}
    monkeypatch.setattr(module, 'RUN', tmp_path)
    monkeypatch.setattr(module, 'load_run', lambda: (protocol, frame))
    monkeypatch.setenv('TFC_API_KEY', 'test-key')
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429)
    client = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    status = module.run_remote()
    assert status['status'] == 'blocked_api'
    assert status['completed'] == 0
    assert len(calls) == 1
    assert 'test-key' not in (tmp_path/'status.json').read_text()
