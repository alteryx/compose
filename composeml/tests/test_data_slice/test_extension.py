from pytest import fixture, mark, raises

from composeml import LabelMaker


@fixture
def data_slice(transactions):
    lm = LabelMaker(
        target_dataframe_name="customer_id", time_index="time", window_size="1h"
    )
    ds = next(lm.slice(transactions, num_examples_per_instance=1))
    return ds


def test_context(data_slice):
    print(data_slice.context)
    context = str(data_slice.context)
    actual = context.splitlines()

    expected = [
        "customer_id                       0",
        "slice_number                      1",
        "slice_start     2019-01-01 08:00:00",
        "slice_stop      2019-01-01 09:00:00",
        "next_start      2019-01-01 09:00:00",
    ]

    assert actual == expected


def test_context_aliases(data_slice):
    assert data_slice.context == data_slice.ctx
    assert data_slice.context.slice_number == data_slice.ctx.count
    assert data_slice.context.slice_start == data_slice.ctx.start
    assert data_slice.context.slice_stop == data_slice.ctx.stop


@mark.parametrize(
    "time_based,offsets",
    argvalues=[
        [False, (2, 4, 2)],
        [False, (2, -6, 2)],
        [True, ("1h", "2h", "1h")],
        [True, ("1h", "-2h30min", "1h")],
        [True, ("2019-01-01 09:00:00", "2019-01-01 10:00:00", "1h")],
    ],
)
def test_subscriptable_slices(transactions, time_based, offsets):
    if time_based:
        dtypes = {"time": "datetime64[ns]"}
        transactions = transactions.astype(dtypes)
        transactions.set_index("time", inplace=True)

    start, stop, size = offsets
    slices = transactions.slice[start:stop:size]
    actual = tuple(map(len, slices))
    assert actual == (2, 2)


def test_subscriptable_error(transactions):
    with raises(TypeError, match="must be a slice object"):
        transactions.slice[0]


def test_time_index_error(transactions):
    match = "offset by frequency requires a time index"
    with raises(AssertionError, match=match):
        transactions.slice[::"1h"]


def test_minimum_data_per_group(transactions):
    lm = LabelMaker(
        "customer_id", labeling_function=len, time_index="time", window_size="1h"
    )
    minimum_data = {1: "2019-01-01 09:00:00", 3: "2019-01-01 12:00:00"}
    lengths = [len(ds) for ds in lm.slice(transactions, 1, minimum_data=minimum_data)]
    assert lengths == [2, 1]


def test_drop_empty(transactions):
    df = transactions.astype({"time": "datetime64[ns]"})
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    ds = df.slice(
        size="1h",
        drop_empty=True,
        stop="2019-01-01 15:00:00",
        start="2019-01-01 08:00:00",
    )

    assert len(list(ds)) == 5
