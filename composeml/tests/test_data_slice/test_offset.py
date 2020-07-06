from composeml.data_slice.offset import DataSliceOffset


def test_numeric_typecast():
    assert int(DataSliceOffset('1 nanosecond')) == 1
    assert float(DataSliceOffset('1970-01-01')) == 0.0
