import pandas as pd


def test_bins(labels):
    given_labels = labels.bin(2)
    transform = given_labels.transforms[0]

    assert transform['transform'] == 'bin'
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

    answer = pd.Categorical(answer, ordered=True)
    labels = labels.assign(my_labeling_function=answer)
    pd.testing.assert_frame_equal(given_labels, labels)


def test_quantile_bins(labels):
    given_labels = labels.bin(2, quantiles=True)
    transform = given_labels.transforms[0]

    assert transform['transform'] == 'bin'
    assert transform['bins'] == 2
    assert transform['quantiles'] is True
    assert transform['labels'] is None
    assert transform['right'] is True

    answer = [
        pd.Interval(137.44, 283.46, closed='right'),
        pd.Interval(31.538999999999998, 137.44, closed='right'),
        pd.Interval(137.44, 283.46, closed='right'),
        pd.Interval(31.538999999999998, 137.44, closed='right'),
    ]

    answer = pd.Categorical(answer, ordered=True)
    labels = labels.assign(my_labeling_function=answer)
    pd.testing.assert_frame_equal(given_labels, labels)
