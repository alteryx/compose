from pytest import raises


def test_threshold(labels):
    labels = labels.threshold(200)
    transform = labels.transforms[0]

    assert transform["transform"] == "threshold"
    assert transform["value"] == 200

    answer = [True, False, True, False]
    target_column = labels.target_columns[0]
    given_answer = labels[target_column].values.tolist()
    assert given_answer == answer


def test_single_target(total_spent):
    lt = total_spent.copy()
    lt.target_columns.append("target_2")
    match = "must first select an individual target"
    with raises(AssertionError, match=match):
        lt.threshold(200)
