import pandas as pd


def test_threshold(labels):
    given_labels = labels.threshold(200)
    transform = given_labels.transforms[0]

    assert transform['__name__'] == 'threshold'
    assert transform['value'] == 200

    answer = [True, False, True, False]
    labels = labels.assign(my_labeling_function=answer)
    pd.testing.assert_frame_equal(given_labels, labels)
