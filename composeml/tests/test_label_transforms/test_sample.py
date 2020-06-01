import pytest

from composeml.tests.utils import to_csv


@pytest.fixture
def labels(labels):
    return labels.threshold(100)


def test_sample_n_int(labels):
    given_answer = labels.sample(n=2, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        'label_id,customer_id,time,my_labeling_function',
        '2,2,2014-01-01 00:01:00,True',
        '3,2,2014-01-01 00:04:00,False',
    ]

    assert given_answer == answer


def test_sample_n_per_label(labels):
    n = {True: 1, False: 2}
    given_answer = labels.sample(n=n, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        'label_id,customer_id,time,my_labeling_function',
        '1,1,2014-01-01 00:48:00,False',
        '2,2,2014-01-01 00:01:00,True',
        '3,2,2014-01-01 00:04:00,False',
    ]

    assert given_answer == answer


def test_sample_frac_int(labels):
    given_answer = labels.sample(frac=.25, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        'label_id,customer_id,time,my_labeling_function',
        '2,2,2014-01-01 00:01:00,True',
    ]

    assert given_answer == answer


def test_sample_frac_per_label(labels):
    frac = {True: 1., False: .5}
    given_answer = labels.sample(frac=frac, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        'label_id,customer_id,time,my_labeling_function',
        '0,1,2014-01-01 00:45:00,True',
        '2,2,2014-01-01 00:01:00,True',
        '3,2,2014-01-01 00:04:00,False',
    ]

    assert given_answer == answer


def test_sample_in_transforms(labels):
    n = {True: 2, False: 2}

    transform = {
        'transform': 'sample',
        'n': n,
        'frac': None,
        'random_state': None,
        'replace': False,
    }

    sample = labels.sample(n=n)
    assert transform != labels.transforms[-1]
    assert transform == sample.transforms[-1]


def test_sample_with_replacement(labels):
    assert labels.shape[0] < 20
    n = {True: 10, False: 10}
    sample = labels.sample(n=n, replace=True)
    assert sample.shape[0] == 20
