import pandas as pd
import pytest

from composeml import LabelTimes
from composeml.tests.utils import read_csv


@pytest.fixture(scope="session")
def transactions():
    df = read_csv(data=[
        'time,amount,customer_id',
        '2019-01-01 08:00:00,1,0',
        '2019-01-01 08:30:00,1,0',
        '2019-01-01 09:00:00,1,1',
        '2019-01-01 09:30:00,1,1',
        '2019-01-01 10:00:00,1,1',
        '2019-01-01 10:30:00,1,2',
        '2019-01-01 11:00:00,1,2',
        '2019-01-01 11:30:00,1,2',
        '2019-01-01 12:00:00,1,2',
        '2019-01-01 12:30:00,1,3',
    ])
    return df


@pytest.fixture(scope="session")
def total_spent_fn():
    def total_spent(df):
        value = df.amount.sum()
        return value

    return total_spent


@pytest.fixture(scope="session")
def unique_amounts_fn():
    def unique_amounts(df):
        return df.amount.nunique()

    return unique_amounts


@pytest.fixture
def total_spent():
    data = [
        'customer_id,time,total_spent',
        '0,2019-01-01 08:00:00,9',
        '0,2019-01-01 08:30:00,8',
        '1,2019-01-01 09:00:00,7',
        '1,2019-01-01 09:30:00,6',
        '1,2019-01-01 10:00:00,5',
        '2,2019-01-01 10:30:00,4',
        '2,2019-01-01 11:00:00,3',
        '2,2019-01-01 11:30:00,2',
        '2,2019-01-01 12:00:00,1',
        '3,2019-01-01 12:30:00,0',
    ]

    data = read_csv(data, parse_dates=['time'])

    label_times = {
        'data': data,
        'settings': {
            'num_examples_per_instance': -1,
        }
    }

    label_times = LabelTimes(**label_times)
    label_times.label_name = 'total_spent'
    label_times.target_entity = 'customer_id'
    return label_times


@pytest.fixture
def labels():
    records = [
        {
            'label_id': 0,
            'customer_id': 1,
            'time': '2014-01-01 00:45:00',
            'my_labeling_function': 226.92999999999998
        },
        {
            'label_id': 1,
            'customer_id': 1,
            'time': '2014-01-01 00:48:00',
            'my_labeling_function': 47.95
        },
        {
            'label_id': 2,
            'customer_id': 2,
            'time': '2014-01-01 00:01:00',
            'my_labeling_function': 283.46000000000004
        },
        {
            'label_id': 3,
            'customer_id': 2,
            'time': '2014-01-01 00:04:00',
            'my_labeling_function': 31.54
        },
    ]

    dtype = {'time': 'datetime64[ns]'}
    values = pd.DataFrame(records).astype(dtype).set_index('label_id')
    values = values[['customer_id', 'time', 'my_labeling_function']]
    values = LabelTimes(values, name='my_labeling_function', target_entity='customer_id')
    return values


@pytest.fixture(autouse=True)
def add_labels(doctest_namespace, labels):
    doctest_namespace['labels'] = labels
