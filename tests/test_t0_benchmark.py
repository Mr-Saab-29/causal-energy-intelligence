import json

import numpy as np
import pandas as pd
import pytest

from src.benchmarks.t0 import build_request, parse_response, prepare, pilot, slices


@pytest.fixture
def sample():
    config = {'targets': ['load'], 'weather': ['temperature'], 'horizon_hours': 24,
              'history_forward_fill_limit_hours': 2, 'quantiles': [0.1, 0.5, 0.9], 'model': 't0-alpha'}
    index = pd.date_range('2026-01-01', periods=24 * 10, freq='h', tz='UTC')
    frame = pd.DataFrame({'load': np.arange(len(index), dtype=float), 'temperature': 10.0}, index=index)
    return frame, index[24 * 7], config


def test_future_targets_never_enter_request(sample):
    frame, origin, config = sample
    before = build_request(frame, origin, 7, config, ['load'], 'historical_weather')
    frame.loc[origin:, :] = 999999
    after = build_request(frame, origin, 7, config, ['load'], 'historical_weather')
    assert before == after
    assert len(before['series'][0]['future_variables']['day_of_week']) == 192
    assert pd.Timestamp(before['series'][0]['index'][-1]) < origin


def test_only_history_is_forward_filled(sample):
    frame, origin, config = sample
    frame.loc[origin - pd.Timedelta(hours=1):origin, 'load'] = np.nan
    _, history, labels = slices(frame, origin, 7, config)
    assert history['load'].iloc[-1] == history['load'].iloc[-2]
    assert pd.isna(labels['load'].iloc[0])


def test_long_gap_and_leading_gap_are_rejected(sample):
    frame, origin, config = sample
    frame.iloc[10:14, 0] = np.nan
    with pytest.raises(ValueError, match='Insufficient'):
        build_request(frame, origin, 7, config, ['load'], 'target_only')


def response(origin):
    return {'series': [[{'index': [x.isoformat() for x in pd.date_range(origin, periods=24, freq='h')],
                        'prediction': {'mean': [2.] * 24, '0.1': [1.] * 24, '0.5': [2.] * 24, '0.9': [3.] * 24}}]]}


def test_response_contract(sample):
    _, origin, config = sample
    assert len(parse_response(response(origin), origin, ['load'], config)) == 24
    bad = response(origin + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match='timestamps'):
        parse_response(bad, origin, ['load'], config)
    bad = response(origin)
    bad['series'][0][0]['prediction']['0.1'][0] = 4
    with pytest.raises(ValueError, match='Crossed'):
        parse_response(bad, origin, ['load'], config)
    bad = response(origin)
    bad['series'][0][0]['prediction']['mean'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        parse_response(bad, origin, ['load'], config)


def test_snapshot_integrity_and_missing_key(sample, tmp_path, monkeypatch):
    frame, origin, config = sample
    source = tmp_path / 'source.csv'
    frame.rename_axis('timestamp_utc').to_csv(source)
    config.update(source=str(source), evaluation_start=origin.isoformat(), evaluation_days=2,
                  context_days=[7], baseline_status='unverified', weather_status='observed',
                  api_url='https://api.retrocast.com/forecast')
    output = tmp_path / 'snapshot'
    audit = prepare(config, output)
    assert audit['eligible_origins_by_context'] == {'7': 2}
    with pytest.raises(ValueError, match='already exists'):
        prepare(config, output)
    monkeypatch.delenv('TFC_API_KEY', raising=False)
    # Production pilot requires 7/30/90 context, so test integrity before case construction.
    (output / 'dataset.csv').write_text('changed')
    with pytest.raises(ValueError, match='Frozen dataset'):
        pilot(output, True)


def test_live_pilot_caches_and_authenticates(tmp_path, monkeypatch):
    import httpx
    import src.benchmarks.t0 as module

    config = json.loads(open('config/benchmark_t0.json').read())
    config.update(targets=['load'], weather=['temperature'], evaluation_start='2026-04-01T00:00:00Z',
                  evaluation_days=3, source=str(tmp_path / 'input.csv'))
    index = pd.date_range('2026-01-01', '2026-04-03 23:00', freq='h', tz='UTC')
    pd.DataFrame({'timestamp_utc': index, 'load': 100., 'temperature': 10.}).to_csv(config['source'], index=False)
    output = tmp_path / 'snapshot'
    prepare(config, output)
    calls = []

    def handler(request):
        assert request.headers['authorization'] == 'Bearer test-only'
        assert request.url.params['model'] == 't0-alpha'
        payload = json.loads(request.content)
        calls.append(payload)
        origin = pd.Timestamp(payload['series'][0]['index'][-1]) + pd.Timedelta(hours=1)
        return httpx.Response(200, json=response(origin))

    original_client = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv('TFC_API_KEY', 'test-only')
    assert pilot(output, True)['status'] == 'completed'
    assert len(calls) == 4
    assert pilot(output, True)['status'] == 'completed'
    assert len(calls) == 4
    assert len(list(output.glob('*.predictions.csv'))) == 4
    assert all('test-only' not in p.read_text() for p in output.glob('*.json'))


def test_duplicate_timestamps_rejected(tmp_path):
    from src.benchmarks.t0 import load_frame
    path = tmp_path / 'bad.csv'
    path.write_text('timestamp_utc,load\n2026-01-01T00:00:00Z,1\n2026-01-01T00:00:00Z,2\n')
    with pytest.raises(ValueError, match='unique hourly'):
        load_frame(path)
