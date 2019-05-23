import pytest
import pandas as pd
from ..label_maker import LabelMaker
from ..label_times import LabelTimes


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
    df = df.set_index('transaction_time')

    return df


@pytest.fixture
def labels():
    records = [
        {
            'customer_id': 1,
            'time': '2014-01-01 00:45:00',
            'my_labeling_function': 226.92999999999998
        },
        {
            'customer_id': 1,
            'time': '2014-01-01 00:48:00',
            'my_labeling_function': 47.95
        },
        {
            'customer_id': 2,
            'time': '2014-01-01 00:01:00',
            'my_labeling_function': 283.46000000000004
        },
        {
            'customer_id': 2,
            'time': '2014-01-01 00:04:00',
            'my_labeling_function': 31.54
        },
    ]

    dtype = {'time': 'datetime64[ns]'}
    df = pd.DataFrame.from_records(records).astype(dtype)
    df = df.set_index(['customer_id', 'time'])

    labels = LabelTimes(df)
    labels.settings.update(name='my_labeling_function')

    return labels


def test_search_by_time(transactions, labels):
    def my_labeling_function(df_slice):
        label = df_slice['amount'].sum()
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
        num_examples_per_instance=2,
        gap='3min',
    )

    assert given_labels.equals(labels)
