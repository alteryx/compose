import pandas as pd
import pytest

from ..label_times import LabelTimes


@pytest.fixture
def labels():
    records = [
        {
            'label_id': 0,
            'customer_id': 1,
            'time': '2014-01-01 00:45:00',
            'my_labeling_function': 226.92999999999998
        },
        {
            'label_id': 1,
            'customer_id': 1,
            'time': '2014-01-01 00:48:00',
            'my_labeling_function': 47.95
        },
        {
            'label_id': 2,
            'customer_id': 2,
            'time': '2014-01-01 00:01:00',
            'my_labeling_function': 283.46000000000004
        },
        {
            'label_id': 3,
            'customer_id': 2,
            'time': '2014-01-01 00:04:00',
            'my_labeling_function': 31.54
        },
    ]

    dtype = {'time': 'datetime64[ns]'}
    df = pd.DataFrame.from_records(records).astype(dtype)

    df = df.set_index('label_id')
    df = df[['customer_id', 'time', 'my_labeling_function']]

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

    answer = [
        '2014-01-01 00:35:00',
        '2014-01-01 00:38:00',
        '2013-12-31 23:51:00',
        '2013-12-31 23:54:00',
    ]

    time = pd.Series(answer, name='time', dtype='datetime64[ns]')
    time = time.rename_axis('label_id')

    pd.testing.assert_series_equal(labels.time, time)


def test_bins(labels):
    given_labels = labels.bin(2)

    answer = [
        pd.Interval(157.5, 283.46, closed='right'),
        pd.Interval(31.288, 157.5, closed='right'),
        pd.Interval(157.5, 283.46, closed='right'),
        pd.Interval(31.288, 157.5, closed='right'),
    ]

    labels['my_labeling_function'] = pd.Categorical(answer, ordered=True)
    pd.testing.assert_frame_equal(given_labels, labels)


def test_quantile_bins(labels):
    given_labels = labels.bin(2, quantiles=True)

    answer = [
        pd.Interval(137.44, 283.46, closed='right'),
        pd.Interval(31.538999999999998, 137.44, closed='right'),
        pd.Interval(137.44, 283.46, closed='right'),
        pd.Interval(31.538999999999998, 137.44, closed='right'),
    ]

    labels['my_labeling_function'] = pd.Categorical(answer, ordered=True)
    pd.testing.assert_frame_equal(given_labels, labels)


def test_describe(labels):
    assert labels.describe() is None


def test_distribution_plot(labels):
    labels = labels.threshold(200)
    plot = labels.plot.distribution()
    assert plot.get_title() == 'label_distribution'


def test_count_by_time_plot(labels):
    labels = labels.threshold(200)
    plot = labels.plot.count_by_time()
    assert plot.get_title() == 'count_by_time'
