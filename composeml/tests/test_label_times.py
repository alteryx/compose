import pandas as pd


def test_threshold(labels):
    given_labels = labels.threshold(200)
    answer = [True, False, True, False]

    labels = labels.copy()
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

    cutoff_time = pd.Series(answer, name='cutoff_time', dtype='datetime64[ns]')
    cutoff_time = cutoff_time.rename_axis('label_id')

    pd.testing.assert_series_equal(labels['cutoff_time'], cutoff_time)


def test_bins(labels):
    given_labels = labels.bin(2)

    answer = [
        pd.Interval(157.5, 283.46, closed='right'),
        pd.Interval(31.288, 157.5, closed='right'),
        pd.Interval(157.5, 283.46, closed='right'),
        pd.Interval(31.288, 157.5, closed='right'),
    ]

    labels = labels.copy()
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

    labels = labels.copy()
    labels['my_labeling_function'] = pd.Categorical(answer, ordered=True)
    pd.testing.assert_frame_equal(given_labels, labels)


def test_describe(labels):
    assert labels.describe() is None
