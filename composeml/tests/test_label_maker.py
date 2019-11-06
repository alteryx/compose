import pandas as pd
import pytest

from composeml import LabelMaker
from composeml.tests.utils import to_csv


def test_search_default(transactions, total_spent_fn):
    lm = LabelMaker(target_entity='customer_id', time_index='time', labeling_function=total_spent_fn)

    given_labels = lm.search(transactions, num_examples_per_instance=1, verbose=False)
    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:00:00,2',
        '1,2019-01-01 09:00:00,3',
        '2,2019-01-01 10:30:00,4',
        '3,2019-01-01 12:30:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_0(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
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

    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:30:00,1',
        '1,2019-01-01 09:30:00,2',
        '2,2019-01-01 11:00:00,3',
    ]

    assert given_labels == labels


def test_search_offset_mix_1(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size=4,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data='2019-01-01 10:00:00',
        gap='4h',
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 10:00:00,4',
        '3,2019-01-01 10:00:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_2(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size='30min',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data=2,
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 11:30:00,1',
        '2,2019-01-01 12:00:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_3(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size='8h',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=-1,
        minimum_data='2019-01-01 08:00:00',
        gap=1,
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

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


def test_search_offset_mix_4(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size=1,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        gap='30min',
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

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


def test_search_offset_mix_5(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data='1h',
        gap=2,
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '1,2019-01-01 10:00:00,1',
        '2,2019-01-01 11:30:00,2',
    ]

    assert given_labels == labels


def test_search_offset_mix_6(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size='1h',
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=1,
        minimum_data=3,
        gap=1,
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '2,2019-01-01 12:00:00,1',
    ]

    assert given_labels == labels


def test_search_offset_mix_7(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (relative).
    """

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size=10,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=float('inf'),
        verbose=False,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        'customer_id,cutoff_time,total_spent',
        '0,2019-01-01 08:00:00,2',
        '1,2019-01-01 09:00:00,3',
        '2,2019-01-01 10:30:00,4',
        '3,2019-01-01 12:30:00,1',
    ]

    assert given_labels == labels


def test_search_offset_negative_0(transactions, total_spent_fn):
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


def test_search_offset_negative_1(transactions, total_spent_fn):
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


def test_invalid_offset(transactions, total_spent_fn):
    match = 'invalid offset'

    with pytest.raises(AssertionError, match=match):
        LabelMaker(
            target_entity='customer_id',
            time_index='time',
            labeling_function=lambda: None,
            window_size={},
        )


def test_invalid_offset_alias(transactions, total_spent_fn):
    match = 'offset must be a valid string'

    with pytest.raises(AssertionError, match=match):
        LabelMaker(
            target_entity='customer_id',
            time_index='time',
            labeling_function=lambda: None,
            window_size='not an offset alias',
        )


def test_invalid_threshold(transactions, total_spent_fn):
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
            verbose=False,
        )


def test_search_invalid_n_examples(transactions, total_spent_fn):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
    )

    with pytest.raises(AssertionError, match='must specify gap'):
        next(lm.slice(transactions, num_examples_per_instance=2, verbose=False))

    with pytest.raises(AssertionError, match='must specify gap'):
        lm.search(transactions, num_examples_per_instance=2, verbose=False)


def test_search_empty_labels(transactions, total_spent_fn):
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
        verbose=False,
    )

    assert given_labels.empty


def test_slice_overlap(transactions, total_spent_fn):
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='time',
        labeling_function=total_spent_fn,
        window_size='1h',
    )

    slices = lm.slice(transactions, num_examples_per_instance=2, verbose=True)

    for df in slices:
        start, end = df.context.window
        is_overlap = df.index == end
        assert not is_overlap.any()


def test_label_type(transactions, total_spent_fn):
    lm = LabelMaker(target_entity='customer_id', time_index='time', labeling_function=total_spent_fn)
    lt = lm.search(transactions, num_examples_per_instance=1, label_type='discrete', verbose=False)
    assert lt.label_type == 'discrete'
