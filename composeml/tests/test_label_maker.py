import pandas as pd
import pytest

from ..label_maker import LabelMaker


def my_labeling_function(df_slice):
    label = df_slice['amount'].sum()
    label = label or pd.np.nan
    return label


def test_search_offset_mix_0(transactions, labels):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (absolute).
    """
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


def test_search_offset_mix_1(transactions, labels):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        minimum_data='1min',
        num_examples_per_instance=4,
        gap='3min',
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_offset_mix_2(transactions, labels):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size='2min',
    )

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=4,
        gap='3min',
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_offset_mix_3(transactions, labels):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (relative).
    """
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
        gap=3,
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_offset_mix_4(transactions, labels):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=4,
        gap='3min',
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_offset_mix_5(transactions, labels):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        minimum_data='1min',
        num_examples_per_instance=4,
        gap=3,
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_offset_mix_6(transactions, labels):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (relative).
    """
    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=my_labeling_function,
        window_size='2min',
    )

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=4,
        gap=3,
    )

    pd.testing.assert_frame_equal(given_labels, labels)


def test_search_offset_mix_7(transactions, labels):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (relative).
    """
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


def test_search_offset_negative_0(transactions):
    match = 'negative offset'

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=None,
        window_size=2,
    )

    with pytest.raises(AssertionError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data=-1,
            gap=-1,
        )


def test_search_offset_negative_1(transactions):
    match = 'negative offset'

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=None,
        window_size=2,
    )

    with pytest.raises(AssertionError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data='-1h',
            gap='-1h',
        )


def test_search_offset_invalid_type(transactions):
    match = 'invalid offset type'

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=None,
        window_size=2,
    )

    with pytest.raises(TypeError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data=[],
            gap=[],
        )


def test_search_empty_labels(transactions):
    transactions['transaction_time'] = pd.NaT

    lm = LabelMaker(
        target_entity='customer_id',
        time_index='transaction_time',
        labeling_function=type(None),
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=2,
        gap=3,
    )

    assert given_labels.empty
