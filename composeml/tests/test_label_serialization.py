import os
import shutil

import pandas as pd
import pytest

import composeml as cp


@pytest.fixture
def path():
    pwd = os.path.dirname(__file__)
    path = os.path.join(pwd, '.cache')
    yield path
    shutil.rmtree(path)


@pytest.fixture
def total_spent(transactions, total_spent_fn):
    lm = cp.LabelMaker(target_entity='customer_id', time_index='time', labeling_function=total_spent_fn)
    lt = lm.search(transactions, num_examples_per_instance=1, verbose=False)
    return lt


def test_csv(path, total_spent):
    total_spent.to_csv(path)
    total_spent_copy = cp.read_csv(path)
    pd.testing.assert_frame_equal(total_spent, total_spent_copy)
    assert total_spent.equals(total_spent_copy)


def test_parquet(path, total_spent):
    total_spent.to_parquet(path)
    total_spent_copy = cp.read_parquet(path)
    pd.testing.assert_frame_equal(total_spent, total_spent_copy)
    assert total_spent.equals(total_spent_copy)


def test_pickle(path, total_spent):
    total_spent.to_pickle(path)
    total_spent_copy = cp.read_pickle(path)
    pd.testing.assert_frame_equal(total_spent, total_spent_copy)
    assert total_spent.equals(total_spent_copy)
