import pandas as pd
import pytest

from .label_times import LabelTimes


@pytest.fixture(scope="module")
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

    df = pd.DataFrame.from_records(records)
    return df


@pytest.fixture(scope="module")
def labels():
    records = [
        {
            'label_id': 0,
            'customer_id': 1,
            'cutoff_time': '2014-01-01 00:45:00',
            'my_labeling_function': 226.92999999999998
        },
        {
            'label_id': 1,
            'customer_id': 1,
            'cutoff_time': '2014-01-01 00:48:00',
            'my_labeling_function': 47.95
        },
        {
            'label_id': 2,
            'customer_id': 2,
            'cutoff_time': '2014-01-01 00:01:00',
            'my_labeling_function': 283.46000000000004
        },
        {
            'label_id': 3,
            'customer_id': 2,
            'cutoff_time': '2014-01-01 00:04:00',
            'my_labeling_function': 31.54
        },
    ]

    dtype = {'cutoff_time': 'datetime64[ns]'}
    values = pd.DataFrame(records).astype(dtype).set_index('label_id')
    values = values[['customer_id', 'cutoff_time', 'my_labeling_function']]
    values = LabelTimes(values, name='my_labeling_function', target_entity='customer_id')
    return values


@pytest.fixture(autouse=True)
def add_labels(doctest_namespace, labels):
    doctest_namespace['labels'] = labels
