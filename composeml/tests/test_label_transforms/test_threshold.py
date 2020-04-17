def test_threshold(labels):
    labels = labels.threshold(200)
    transform = labels.transforms[0]

    assert transform['transform'] == 'threshold'
    assert transform['value'] == 200

    answer = [True, False, True, False]
    given_answer = labels[labels.label_name].values.tolist()
    assert given_answer == answer
