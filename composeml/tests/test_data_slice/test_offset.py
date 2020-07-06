from composeml.data_slice.offset import DataSliceOffset
from pytest import raises


def test_numeric_typecast():
    assert int(DataSliceOffset('1 nanosecond')) == 1
    assert float(DataSliceOffset('1970-01-01')) == 0.0


def test_invalid_value():
    with raises(AssertionError, match='invalid offset'):
        DataSliceOffset(None)


def test_alias_phrase():
    phrase = 'until start of next month'
    actual = DataSliceOffset(phrase).value
    expected = DataSliceOffset('1MS').value
    assert actual == expected