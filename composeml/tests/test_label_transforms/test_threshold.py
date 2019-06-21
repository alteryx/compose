import pandas as pd


def test_threshold(labels):
    labels = labels.copy()
    given_labels = labels.threshold(200)

    answer = [True, False, True, False]
    labels['my_labeling_function'] = answer

    pd.testing.assert_frame_equal(given_labels, labels)
    assert given_labels.settings.get('threshold') == 200
