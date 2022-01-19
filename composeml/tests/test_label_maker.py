import pandas as pd
import pytest

from composeml import LabelMaker
from composeml.tests.utils import to_csv


def test_search_default(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
    )

    given_labels = lm.search(transactions, num_examples_per_instance=1)
    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,2",
        "1,2019-01-01 09:00:00,3",
        "2,2019-01-01 10:30:00,4",
        "3,2019-01-01 12:30:00,1",
    ]

    assert given_labels == labels


def test_search_examples_per_label(transactions, total_spent_fn):
    def total_spent(ds):
        return total_spent_fn(ds) > 2

    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent,
    )

    n_examples = {True: -1, False: 1}
    given_labels = lm.search(transactions, num_examples_per_instance=n_examples, gap=1)
    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,False",
        "1,2019-01-01 09:00:00,True",
        "1,2019-01-01 09:30:00,False",
        "2,2019-01-01 10:30:00,True",
        "2,2019-01-01 11:00:00,True",
        "2,2019-01-01 11:30:00,False",
        "3,2019-01-01 12:30:00,False",
    ]

    assert given_labels == labels


def test_search_with_undefined_labels(transactions, total_spent_fn):
    def total_spent(ds):
        return total_spent_fn(ds) % 3

    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent,
    )

    n_examples = {1: 1, 2: 1}
    given_labels = lm.search(transactions, num_examples_per_instance=n_examples, gap=1)
    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,2",
        "0,2019-01-01 08:30:00,1",
        "1,2019-01-01 09:30:00,2",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 10:30:00,1",
        "2,2019-01-01 11:30:00,2",
        "3,2019-01-01 12:30:00,1",
    ]

    assert given_labels == labels


def test_search_with_multiple_targets(transactions, total_spent_fn, unique_amounts_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        window_size=2,
        labeling_function={
            "total_spent": total_spent_fn,
            "unique_amounts": unique_amounts_fn,
        },
    )

    expected = [
        "customer_id,time,total_spent,unique_amounts",
        "0,2019-01-01 08:00:00,2,1",
        "1,2019-01-01 09:00:00,2,1",
        "1,2019-01-01 10:00:00,1,1",
        "2,2019-01-01 10:30:00,2,1",
        "2,2019-01-01 11:30:00,2,1",
        "3,2019-01-01 12:30:00,1,1",
    ]

    lt = lm.search(transactions, num_examples_per_instance=-1)
    actual = lt.pipe(to_csv, index=False)
    info = "unexpected calculated values"
    assert actual == expected, info

    expected = [
        "customer_id,time,unique_amounts",
        "0,2019-01-01 08:00:00,1",
        "1,2019-01-01 09:00:00,1",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 10:30:00,1",
        "2,2019-01-01 11:30:00,1",
        "3,2019-01-01 12:30:00,1",
    ]

    actual = lt.select("unique_amounts")
    actual = actual.pipe(to_csv, index=False)
    info = "selected values differ from calculated values"
    assert actual == expected, info


def test_search_offset_mix_0(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size="2h",
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data="30min",
        gap="2h",
        drop_empty=True,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:30:00,1",
        "1,2019-01-01 09:30:00,2",
        "2,2019-01-01 11:00:00,3",
    ]

    assert given_labels == labels


def test_search_offset_mix_1(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (absolute).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size=4,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data="2019-01-01 10:00:00",
        gap="4h",
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 10:00:00,4",
        "3,2019-01-01 10:00:00,1",
    ]

    assert given_labels == labels


def test_search_offset_mix_2(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size="30min",
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data=2,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 11:30:00,1",
        "2,2019-01-01 12:00:00,1",
    ]

    assert given_labels == labels


def test_search_offset_mix_3(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size="8h",
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=-1,
        minimum_data="2019-01-01 08:00:00",
        gap=1,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,2",
        "0,2019-01-01 08:30:00,1",
        "1,2019-01-01 09:00:00,3",
        "1,2019-01-01 09:30:00,2",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 10:30:00,4",
        "2,2019-01-01 11:00:00,3",
        "2,2019-01-01 11:30:00,2",
        "2,2019-01-01 12:00:00,1",
        "3,2019-01-01 12:30:00,1",
    ]

    assert given_labels == labels


def test_search_offset_mix_4(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (absolute).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size=1,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        gap="30min",
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,1",
        "0,2019-01-01 08:30:00,1",
        "1,2019-01-01 09:00:00,1",
        "1,2019-01-01 09:30:00,1",
        "2,2019-01-01 10:30:00,1",
        "2,2019-01-01 11:00:00,1",
        "3,2019-01-01 12:30:00,1",
    ]

    assert given_labels == labels


def test_search_offset_mix_5(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (absolute), and gap (relative).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=2,
        minimum_data="1h",
        gap=2,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 11:30:00,2",
    ]

    assert given_labels == labels


def test_search_offset_mix_6(transactions, total_spent_fn):
    """
    Test offset mix with window_size (absolute), minimum_data (relative), and gap (relative).
    """
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size="1h",
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=1,
        minimum_data=3,
        gap=1,
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "2,2019-01-01 12:00:00,1",
    ]

    assert given_labels == labels


def test_search_offset_mix_7(transactions, total_spent_fn):
    """
    Test offset mix with window_size (relative), minimum_data (relative), and gap (relative).
    """

    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size=10,
    )

    given_labels = lm.search(
        transactions,
        num_examples_per_instance=float("inf"),
    )

    given_labels = to_csv(given_labels, index=False)

    labels = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,2",
        "1,2019-01-01 09:00:00,3",
        "2,2019-01-01 10:30:00,4",
        "3,2019-01-01 12:30:00,1",
    ]

    assert given_labels == labels


def test_search_offset_negative_0(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=lambda: None,
        window_size=2,
    )

    match = "offset must be positive"
    with pytest.raises(AssertionError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data=-1,
            gap=-1,
        )


def test_search_offset_negative_1(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=lambda: None,
        window_size=2,
    )

    match = "offset must be positive"
    with pytest.raises(AssertionError, match=match):
        lm.search(
            transactions,
            num_examples_per_instance=2,
            minimum_data="-1h",
            gap="-1h",
        )


def test_search_invalid_n_examples(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
    )

    with pytest.raises(AssertionError, match="must specify gap"):
        next(lm.slice(transactions, num_examples_per_instance=2))

    with pytest.raises(AssertionError, match="must specify gap"):
        lm.search(transactions, num_examples_per_instance=2)


def test_column_based_windows(transactions, total_spent_fn):
    session_id = [1, 2, 3, 3, 4, 5, 5, 5, 6, 7]
    df = transactions.assign(session_id=session_id)

    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        window_size="session_id",
        labeling_function=total_spent_fn,
    )

    actual = lm.search(df, -1).pipe(to_csv, index=False)

    expected = [
        "customer_id,time,total_spent",
        "0,2019-01-01 08:00:00,1",
        "0,2019-01-01 08:30:00,1",
        "1,2019-01-01 09:00:00,2",
        "1,2019-01-01 10:00:00,1",
        "2,2019-01-01 10:30:00,3",
        "2,2019-01-01 12:00:00,1",
        "3,2019-01-01 12:30:00,1",
    ]

    assert actual == expected


def test_search_with_invalid_index(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=lambda df: None,
        window_size=2,
    )

    df = transactions.sample(n=10, random_state=0)
    match = "data frame must be sorted chronologically"
    with pytest.raises(AssertionError, match=match):
        lm.search(df, num_examples_per_instance=2)

    df = transactions.assign(time=pd.NaT)
    match = "index contains null values"
    with pytest.raises(AssertionError, match=match):
        lm.search(df, num_examples_per_instance=2)


def test_search_on_empty_labels(transactions):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=lambda ds: None,
        window_size=2,
    )

    given_labels = lm.search(
        transactions,
        minimum_data=1,
        num_examples_per_instance=2,
        gap=1,
    )

    assert given_labels.empty


def test_data_slice_overlap(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
        window_size="1h",
    )

    for ds in lm.slice(transactions, num_examples_per_instance=2):
        overlap = ds.index == ds.context.slice_stop
        assert not overlap.any()


def test_label_type(transactions, total_spent_fn):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=total_spent_fn,
    )
    lt = lm.search(transactions, num_examples_per_instance=1)
    assert lt.target_types["total_spent"] == "continuous"
    assert lt.bin(2).target_types["total_spent"] == "discrete"


def test_search_with_maximum_data(transactions):
    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="time",
        labeling_function=len,
        window_size="1h",
    )

    lt = lm.search(
        df=transactions.sort_values("time"),
        num_examples_per_instance=-1,
        minimum_data="2019-01-01 08:00:00",
        maximum_data="2019-01-01 09:00:00",
        drop_empty=False,
    )

    expected = [
        "customer_id,time,len",
        "0,2019-01-01 08:00:00,2",
        "0,2019-01-01 09:00:00,0",
        "1,2019-01-01 08:00:00,0",
        "1,2019-01-01 09:00:00,2",
        "2,2019-01-01 08:00:00,0",
        "2,2019-01-01 09:00:00,0",
        "3,2019-01-01 08:00:00,0",
        "3,2019-01-01 09:00:00,0",
    ]

    actual = lt.pipe(to_csv, index=False)
    assert actual == expected

    lt = lm.search(
        df=transactions.sort_values("time"),
        num_examples_per_instance=-1,
        maximum_data="30min",
        drop_empty=False,
        gap="30min",
    )

    expected = [
        "customer_id,time,len",
        "0,2019-01-01 08:00:00,2",
        "0,2019-01-01 08:30:00,1",
        "1,2019-01-01 09:00:00,2",
        "1,2019-01-01 09:30:00,2",
        "2,2019-01-01 10:30:00,2",
        "2,2019-01-01 11:00:00,2",
        "3,2019-01-01 12:30:00,1",
        "3,2019-01-01 13:00:00,0",
    ]

    actual = lt.pipe(to_csv, index=False)
    assert actual == expected


@pytest.mark.parametrize(
    "minimum_data",
    [
        {1: "2019-01-01 09:30:00", 2: "2019-01-01 11:30:00"},
        {1: "30min", 2: "1h"},
        {1: 1, 2: 2},
    ],
)
def test_minimum_data_per_group(transactions, minimum_data):
    lm = LabelMaker(
        "customer_id", labeling_function=len, time_index="time", window_size="1h"
    )
    for supported_type in [minimum_data, pd.Series(minimum_data)]:
        lt = lm.search(transactions, 1, minimum_data=supported_type)
        actual = to_csv(lt, index=False)

        expected = [
            "customer_id,time,len",
            "1,2019-01-01 09:30:00,2",
            "2,2019-01-01 11:30:00,2",
        ]

        assert actual == expected


def test_minimum_data_per_group_error(transactions):
    lm = LabelMaker(
        "customer_id", labeling_function=len, time_index="time", window_size="1h"
    )
    data = ["2019-01-01 09:00:00", "2019-01-01 12:00:00"]
    minimum_data = pd.Series(data=data, index=[1, 1])
    match = "more than one cutoff time exists for a target group"

    with pytest.raises(ValueError, match=match):
        lm.search(transactions, 1, minimum_data=minimum_data)
