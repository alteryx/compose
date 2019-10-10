import re

import pandas as pd

from composeml.utils import can_be_type


def parse(value):
    """Maps text to offset.

    Args:
        value (str) : Text description of offset.

    Returns:
        BaseOffset : Instance of offset.
    """
    pattern = re.compile('until start of next (?P<unit>[a-z]+)')
    match = pattern.search(value.lower())

    if match:
        match = match.groupdict()
        unit = match['unit']

        if unit == 'month':
            return pd.offsets.MonthBegin()

        if unit == 'year':
            return pd.offsets.YearBegin()


def is_offset(value):
    """Checks whether a value is an offset.

    Args:
        value (any) : Value to check.

    Returns:
        Bool : Whether value is an offset.
    """
    return issubclass(type(value), pd.tseries.offsets.BaseOffset)


def to_offset(value):
    """Converts a value to an offset and validates the offset.

    Args:
        value (int or str or offset) : Value of offset.

    Returns:
        offset : Valid offset.
    """
    if isinstance(value, int):
        assert value > 0, 'offset must be greater than zero'
        offset = value

    elif isinstance(value, str):
        offset = parse(value)

        if offset is None:
            error = 'offset must be a valid string'
            assert can_be_type(type=pd.tseries.frequencies.to_offset, string=value), error
            offset = pd.tseries.frequencies.to_offset(value)
            assert offset.n > 0, 'offset must be greater than zero'

    else:
        assert is_offset(value), 'invalid offset'
        assert value.n > 0, 'offset must be greater than zero'
        offset = value

    return offset
