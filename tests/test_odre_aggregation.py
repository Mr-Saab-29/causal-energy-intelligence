"""Regression cases for source cadence and incomplete-hour rejection."""
from decimal import Decimal

import pytest

from src.data.sources.odre import (
    ODRE_NATIONAL_DATASET, ODRE_NATIONAL_HISTORICAL_DATASET,
    ODRE_REGIONAL_HISTORICAL_DATASET, aggregate_odre_to_hourly_mwh,
)


def records(minutes, values, **fields):
    return [{'date_heure': f'2026-04-30T00:{minute:02d}:00+00:00',
             'consommation': value, **fields} for minute, value in zip(minutes, values)]


def test_half_hour_actuals_with_empty_quarter_slots():
    result = aggregate_odre_to_hourly_mwh(
        records([0, 15, 30, 45], [41104, None, 40013, None]),
        'national', dataset=ODRE_NATIONAL_HISTORICAL_DATASET)
    assert result[0].consumption_mwh == Decimal('40558.5')
    assert result[0].total_production_mwh is None


def test_quarter_hour_actuals_keep_their_scale():
    result = aggregate_odre_to_hourly_mwh(records([0, 15, 30, 45], [40000] * 4),
                                         'national', dataset=ODRE_NATIONAL_DATASET)
    assert result[0].consumption_mwh == Decimal('40000')


@pytest.mark.parametrize('dataset,minutes', [
    (ODRE_NATIONAL_DATASET, [0, 30]),
    (ODRE_NATIONAL_HISTORICAL_DATASET, [0]),
])
def test_missing_intervals_are_not_reweighted(dataset, minutes):
    result = aggregate_odre_to_hourly_mwh(records(minutes, [40000] * len(minutes)),
                                         'national', dataset=dataset)
    assert result[0].consumption_mwh is None


def test_generation_and_carbon_use_complete_intervals():
    fields = dict(nucleaire=30000, gaz=1000, charbon=0, fioul=0,
                  eolien=5000, solaire=0, hydraulique=3000, bioenergies=1000, taux_co2=20)
    raw = records([0, 30], [40000] * 2, **fields)
    result = aggregate_odre_to_hourly_mwh(raw, 'national', dataset=ODRE_NATIONAL_HISTORICAL_DATASET)[0]
    assert result.total_production_mwh == Decimal('40000')
    assert result.coal_mwh == Decimal('0')
    assert result.carbon_intensity_gco2_kwh == Decimal('20')
    raw[1]['gaz'] = None
    raw[1]['taux_co2'] = None
    result = aggregate_odre_to_hourly_mwh(raw, 'national', dataset=ODRE_NATIONAL_HISTORICAL_DATASET)[0]
    assert result.gas_mwh is None
    assert result.total_production_mwh is None
    assert result.carbon_intensity_gco2_kwh is None


def test_duplicate_and_contradictory_records_fail():
    raw = records([0, 0], [40000, 41000])
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        aggregate_odre_to_hourly_mwh(raw, 'national', dataset=ODRE_NATIONAL_DATASET)
    raw = records([0], [40000], nature='Données consolidées')
    with pytest.raises(ValueError, match='contradicts'):
        aggregate_odre_to_hourly_mwh(raw, 'national', dataset=ODRE_NATIONAL_DATASET)
    with pytest.raises(ValueError, match='Invalid'):
        aggregate_odre_to_hourly_mwh(records([15], [40000]), 'national', dataset=ODRE_NATIONAL_HISTORICAL_DATASET)


def test_regional_half_hour_actuals():
    result = aggregate_odre_to_hourly_mwh(
        records([0, 30], [1000, 1200], libelle_region='Bretagne'),
        'regional', dataset=ODRE_REGIONAL_HISTORICAL_DATASET)
    assert result[0].consumption_mwh == Decimal('1100')


def test_duplicate_actuals_ignore_unused_dst_labels_and_forecasts():
    raw = records([0, 30], [44716, 45124], taux_co2=25)
    duplicate = {**raw[0], 'heure': '03:00', 'prevision_j1': 43600}
    raw[0].update(heure='02:00', prevision_j1=42700)
    result = aggregate_odre_to_hourly_mwh(
        [raw[0], duplicate, raw[1]], 'national', dataset=ODRE_NATIONAL_HISTORICAL_DATASET)
    assert result[0].consumption_mwh == Decimal('44920')
    assert result[0].carbon_intensity_gco2_kwh == Decimal('25')


def test_distinct_utc_instants_with_same_local_label_are_not_duplicates():
    raw = records([0, 30], [40000, 40000], heure='02:00')
    later = [{**row, 'date_heure': row['date_heure'].replace('T00:', 'T01:')} for row in raw]
    result = aggregate_odre_to_hourly_mwh(
        raw + later, 'national', dataset=ODRE_NATIONAL_HISTORICAL_DATASET)
    assert len(result) == 2
    assert all(row.consumption_mwh == Decimal('40000') for row in result)


def test_legacy_benchmark_is_rejected():
    from src.benchmarks.rebuild_energy import verify_energy_source
    with pytest.raises(ValueError, match='Uncorrected'):
        verify_energy_source({'source': 'old.csv'})
