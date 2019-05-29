import pandas as pd
import pytest

from ..label_times import LabelTimes


@pytest.fixture
def labels():
    records = [
        {
            'customer_id': 1,
            'time': '2014-01-01 00:45:00',
            'my_labeling_function': 226.92999999999998
        },
        {
            'customer_id': 1,
            'time': '2014-01-01 00:48:00',
            'my_labeling_function': 47.95
        },
        {
            'customer_id': 2,
            'time': '2014-01-01 00:01:00',
            'my_labeling_function': 283.46000000000004
        },
        {
            'customer_id': 2,
            'time': '2014-01-01 00:04:00',
            'my_labeling_function': 31.54
        },
    ]

    dtype = {'time': 'datetime64[ns]'}
    df = pd.DataFrame.from_records(records).astype(dtype)
    df = df.set_index(['customer_id', 'time'])

    labels = LabelTimes(df)
    labels.settings = {
        'name': 'my_labeling_function',
        'target_entity': 'customer_id',
    }

    return labels


def test_threshold(labels):
    given_labels = labels.threshold(200)
    answer = [True, False, True, False]
    labels['my_labeling_function'] = answer

    pd.testing.assert_frame_equal(given_labels, labels)
    assert given_labels.settings.get('threshold') == 200


def test_lead(labels):
    labels = labels.apply_lead('10min')
    assert labels.settings.get('lead') == '10min'
    given_time = labels.index.get_level_values('time')

    answer = [
        '2014-01-01 00:35:00',
        '2014-01-01 00:38:00',
        '2013-12-31 23:51:00',
        '2013-12-31 23:54:00',
    ]

    time = pd.DatetimeIndex(answer, name='time')
    pd.testing.assert_index_equal(given_time, time)


def test_describe(labels):
    assert labels.describe() is None
