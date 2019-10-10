import pandas as pd

from composeml.offsets import parse


def test_parser():
    offset = parse('until start of next month')
    assert isinstance(offset, pd.offsets.MonthBegin)
    offset = parse('until start of next year')
    assert isinstance(offset, pd.offsets.YearBegin)
