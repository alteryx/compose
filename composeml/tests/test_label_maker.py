import pandas as pd
import pytest

from composeml import LabelMaker
from composeml.tests.utils import read_csv


@pytest.fixture
def transactions():

    data = [
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
    ]

    df = read_csv(data)
    return df


def total_spent(df):
    total = df.amount.sum()
    return total


def test_search_offset_mix_0(transactions):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size='2h',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data='30min',
        gap='2h',
        drop_empty=True,
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:30:00,1',
        '1,2019-01-01 09:30:00,2',
        '2,2019-01-01 11:00:00,3',
    ]

    assert given_labels == labels


def test_search_offset_mix_1(transactions):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size=4,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data='2019-01-01 10:00:00',
        gap='4h',
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 10:00:00,4',
        '3,2019-01-01 10:00:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_2(transactions):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size='30min',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data=2,
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 11:30:00,1',
        '2,2019-01-01 12:00:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_3(transactions):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size='8h',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=-1,
        minimum_data='2019-01-01 08:00:00',
        gap=1,
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:00:00,2',
        '0,2019-01-01 08:30:00,1',
        '1,2019-01-01 09:00:00,3',
        '1,2019-01-01 09:30:00,2',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 10:30:00,4',
        '2,2019-01-01 11:00:00,3',
        '2,2019-01-01 11:30:00,2',
        '2,2019-01-01 12:00:00,1',
        '3,2019-01-01 12:30:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_4(transactions):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size=1,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        gap='30min',
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:00:00,1',
        '0,2019-01-01 08:30:00,1',
        '1,2019-01-01 09:00:00,1',
        '1,2019-01-01 09:30:00,1',
        '2,2019-01-01 10:30:00,1',
        '2,2019-01-01 11:00:00,1',
        '3,2019-01-01 12:30:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_5(transactions):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data='1h',
        gap=2,
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 11:30:00,2',
    ]

    assert given_labels == labels


def test_search_offset_mix_6(transactions):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size='1h',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=1,
        minimum_data=3,
        gap=1,
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '2,2019-01-01 12:00:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_7(transactions):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (relative).
    """

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size=10,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=float('inf'),
        verbose=False,
    )

    given_labels = given_labels.to_csv(index=False).splitlines()

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:00:00,2',
        '1,2019-01-01 09:00:00,3',
        '2,2019-01-01 10:30:00,4',
        '3,2019-01-01 12:30:00,1',
    ]

    assert given_labels == labels


def test_search_offset_negative_0(transactions):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=lambda: None,
        window_size=2,
    )

    match = 'must be greater than zero'

    with pytest.raises(AssertionError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data=-1,
            gap=-1,
            verbose=False,
        )


def test_search_offset_negative_1(transactions):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=lambda: None,
        window_size=2,
    )

    match = 'must be greater than zero'

    with pytest.raises(AssertionError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data='-1h',
            gap='-1h',
            verbose=False,
        )


def test_invalid_offset(transactions):
    match = 'invalid offset'

    with pytest.raises(AssertionError, match=match):
        LabelMaker(
            target_entity='customer_id',
            time_index='time',
            labeling_function=lambda: None,
            window_size={},
        )


def test_invalid_threshold(transactions):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=lambda: None,
        window_size=2,
    )

    match = 'invalid threshold'

    with pytest.raises(ValueError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data=' ',
        )


def test_search_empty_labels(transactions):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=lambda df: None,
        window_size=2,
    )

    transactions = transactions.assign(time=pd.NaT)

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=2,
        gap=3,
    )

    assert given_labels.empty


def test_slice(transactions):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent,
        window_size='2h',
    )

    slices = lm.slice(
        transactions,
        num_examples_per_instance=2,
        minimum_data='30min',
        gap='2h',
        metadata=True,
        verbose=True,
    )

    for df, metadata in slices:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(metadata, dict)
