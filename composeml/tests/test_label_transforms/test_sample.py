import pytest

from composeml import LabelTimes
from composeml.tests.utils import read_csv, to_csv


@pytest.fixture
def labels(labels):
    return labels.threshold(100)


def test_sample_n_int(labels):
    given_answer = labels.sample(n=2, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        "label_id,customer_id,time,my_labeling_function",
        "2,2,2014-01-01 00:01:00,True",
        "3,2,2014-01-01 00:04:00,False",
    ]

    assert given_answer == answer


def test_sample_n_per_label(labels):
    n = {True: 1, False: 2}
    given_answer = labels.sample(n=n, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        "label_id,customer_id,time,my_labeling_function",
        "1,1,2014-01-01 00:48:00,False",
        "2,2,2014-01-01 00:01:00,True",
        "3,2,2014-01-01 00:04:00,False",
    ]

    assert given_answer == answer


def test_sample_frac_int(labels):
    given_answer = labels.sample(frac=0.25, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        "label_id,customer_id,time,my_labeling_function",
        "2,2,2014-01-01 00:01:00,True",
    ]

    assert given_answer == answer


def test_sample_frac_per_label(labels):
    frac = {True: 1.0, False: 0.5}
    given_answer = labels.sample(frac=frac, random_state=0)
    given_answer = given_answer.sort_index()
    given_answer = to_csv(given_answer, index=True)

    answer = [
        "label_id,customer_id,time,my_labeling_function",
        "0,1,2014-01-01 00:45:00,True",
        "2,2,2014-01-01 00:01:00,True",
        "3,2,2014-01-01 00:04:00,False",
    ]

    assert given_answer == answer


def test_sample_in_transforms(labels):
    n = {True: 2, False: 2}

    transform = {
        "transform": "sample",
        "n": n,
        "frac": None,
        "random_state": None,
        "replace": False,
        "per_instance": False,
    }

    sample = labels.sample(n=n)
    assert transform != labels.transforms[-1]
    assert transform == sample.transforms[-1]


def test_sample_with_replacement(labels):
    assert labels.shape[0] < 20
    n = {True: 10, False: 10}
    sample = labels.sample(n=n, replace=True)
    assert sample.shape[0] == 20


def test_single_target(total_spent):
    lt = total_spent.copy()
    lt.target_columns.append("target_2")
    match = "must first select an individual target"
    with pytest.raises(AssertionError, match=match):
        lt.sample(2)


def test_sample_n_per_instance():
    data = read_csv(
        [
            "target_dataframe_name,labels",
            "0,a",
            "0,b",
            "1,a",
            "1,b",
        ]
    )

    lt = LabelTimes(data=data, target_dataframe_name="target_dataframe_name")
    sample = lt.sample(n={"a": 1}, per_instance=True, random_state=0)
    actual = to_csv(sample, index=False)

    expected = [
        "target_dataframe_name,labels",
        "0,a",
        "1,a",
    ]

    assert expected == actual


def test_sample_frac_per_instance():
    data = read_csv(
        [
            "target_dataframe_name,labels",
            "0,a",
            "0,a",
            "0,a",
            "0,a",
            "1,a",
            "1,a",
        ]
    )

    lt = LabelTimes(data=data, target_dataframe_name="target_dataframe_name")
    sample = lt.sample(frac={"a": 0.5}, per_instance=True, random_state=0)
    actual = to_csv(sample, index=False)

    expected = [
        "target_dataframe_name,labels",
        "0,a",
        "0,a",
        "1,a",
    ]

    assert expected == actual
