import pytest


@pytest.fixture
def labels(labels):
    return labels.threshold(100)


def test_sample_n_int(labels):
    given_answer = labels.sample(n=2, random_state=0).sort_index()
    given_answer = given_answer.to_csv(index=True).splitlines()

    answer = [
        'label_id,customer_id,cutoff_time,my_labeling_function',
        '2,2,2014-01-01 00:01:00,True',
        '3,2,2014-01-01 00:04:00,False',
    ]

    assert given_answer == answer


def test_sample_n_dict(labels):
    n = {True: 1, False: 2}
    given_answer = labels.sample(n=n, random_state=0).sort_index()
    given_answer = given_answer.to_csv(index=True).splitlines()

    answer = [
        'label_id,customer_id,cutoff_time,my_labeling_function',
        '1,1,2014-01-01 00:48:00,False',
        '2,2,2014-01-01 00:01:00,True',
        '3,2,2014-01-01 00:04:00,False',
    ]

    assert given_answer == answer


def test_sample_frac_int(labels):
    given_answer = labels.sample(frac=.25, random_state=0).sort_index()
    given_answer = given_answer.to_csv(index=True).splitlines()

    answer = [
        'label_id,customer_id,cutoff_time,my_labeling_function',
        '2,2,2014-01-01 00:01:00,True',
    ]

    assert given_answer == answer


def test_sample_frac_dict(labels):
    frac = {True: 1., False: .5}
    given_answer = labels.sample(frac=frac, random_state=0).sort_index()
    given_answer = given_answer.to_csv(index=True).splitlines()

    answer = [
        'label_id,customer_id,cutoff_time,my_labeling_function',
        '0,1,2014-01-01 00:45:00,True',
        '2,2,2014-01-01 00:01:00,True',
        '3,2,2014-01-01 00:04:00,False',
    ]

    assert given_answer == answer
