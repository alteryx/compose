import pandas as pd


def test_bins(labels):
    labels = labels.copy()
    given_labels = labels.bin(2)

    transform = given_labels.transforms[0]
    assert transform['name'] == 'bin'
    assert transform['bins'] == 2
    assert transform['quantiles'] is False
    assert transform['labels'] is None
    assert transform['right'] is True

    answer = [
        pd.Interval(157.5, 283.46, closed='right'),
        pd.Interval(31.288, 157.5, closed='right'),
        pd.Interval(157.5, 283.46, closed='right'),
        pd.Interval(31.288, 157.5, closed='right'),
    ]

    labels['my_labeling_function'] = pd.Categorical(answer, ordered=True)
    pd.testing.assert_frame_equal(given_labels, labels)


def test_quantile_bins(labels):
    labels = labels.copy()
    given_labels = labels.bin(2, quantiles=True)

    assert given_labels.settings.get('bins') == 2
    assert given_labels.settings.get('quantiles') is True
    assert given_labels.settings.get('labels') is None
    assert given_labels.settings.get('right') is True

    answer = [
        pd.Interval(137.44, 283.46, closed='right'),
        pd.Interval(31.538999999999998, 137.44, closed='right'),
        pd.Interval(137.44, 283.46, closed='right'),
        pd.Interval(31.538999999999998, 137.44, closed='right'),
    ]

    labels['my_labeling_function'] = pd.Categorical(answer, ordered=True)
    pd.testing.assert_frame_equal(given_labels, labels)
