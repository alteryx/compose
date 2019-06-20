import pandas as pd
import pytest

from .label_times import LabelTimes


@pytest.fixture(scope="module")
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
    df = pd.DataFrame.from_records(records).astype(dtype)

    df = df.set_index('label_id')
    df = df[['customer_id', 'time', 'my_labeling_function']]

    labels = LabelTimes(df)
    labels.settings = {
        'name': 'my_labeling_function',
        'target_entity': 'customer_id',
    }

    return labels


@pytest.fixture(autouse=True)
def add_labels(doctest_namespace, labels):
    doctest_namespace['labels'] = labels
