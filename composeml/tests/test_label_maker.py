# flake8:noqa
import pandas as pd
import pytest

from ..label_maker import LabelMaker
from .test_label_times import labels


@pytest.fixture
def transactions():
    records = [
        {
            'transaction_time': '2014-01-01 00:44:00',
            'amount': 21.35,
            'customer_id': 1
        },
        {
            'transaction_time': '2014-01-01 00:45:00',
            'amount': 108.11,
            'customer_id': 1
        },
        {
            'transaction_time': '2014-01-01 00:46:00',
            'amount': 112.53,
            'customer_id': 1
        },
        {
            'transaction_time': '2014-01-01 00:47:00',
            'amount': 6.29,
            'customer_id': 1
        },
        {
            'transaction_time': '2014-01-01 00:48:00',
            'amount': 47.95,
            'customer_id': 1
        },
        {
            'transaction_time': '2014-01-01 00:00:00',
            'amount': 127.64,
            'customer_id': 2
        },
        {
            'transaction_time': '2014-01-01 00:01:00',
            'amount': 109.48,
            'customer_id': 2
        },
        {
            'transaction_time': '2014-01-01 00:02:00',
            'amount': 95.06,
            'customer_id': 2
        },
        {
            'transaction_time': '2014-01-01 00:03:00',
            'amount': 78.92,
            'customer_id': 2
        },
        {
            'transaction_time': '2014-01-01 00:04:00',
            'amount': 31.54,
            'customer_id': 2
        },
    ]

    dtype = {'transaction_time': 'datetime64[ns]'}
    df = pd.DataFrame.from_records(records).astype(dtype)
    return df


def test_search_by_time(transactions, labels):
    def my_labeling_function(df_slice):
        label = df_slice['amount'].sum()
        label = label or pd.np.nan
        return label

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size='2min',
    )

    given_labels = lm.search(
        transactions,
        minimum_data='1min',
        num_examples_per_instance=4,
        gap='3min',
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_by_observations(transactions, labels):
    def my_labeling_function(df_slice):
        label = df_slice['amount'].sum()
        return label

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=2,
        gap=3,
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_invalid_offset(transactions):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=None,
        window_size=2,
    )

    text = 'time offset must be integer or string'
    with pytest.raises(TypeError, match=text):
        given_labels = lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data={},
            gap={},
        )
