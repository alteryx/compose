import re

import pandas as pd


class DataSliceOffset:
    def __init__(self, value):
        self.value = value
        self._check()

    def _check(self):
        if isinstance(self.value, str):
            self.value = self._parse_value()

        info = 'invalid offset'
        assert self._is_valid_offset, info

    def _parse_value(self):
        value = self._alias_phrase_to_offset(self.value)
        value = value or self._alias_to_offset(self.value)
        return value

    @property
    def _is_offset_frequency(self):
        return self._is_offset_timedelta or self._is_offset_base

    @property
    def _is_offset_base(self):
        return issubclass(type(self.value), pd.tseries.offsets.BaseOffset)

    @property
    def _is_offset_position(self):
        return isinstance(self.value, int)

    @property
    def _is_offset_timedelta(self):
        return issubclass(type(self.value), pd.Timedelta)

    @property
    def _is_valid_offset(self):
        return self._is_offset_position or self._is_offset_frequency

    @staticmethod
    def _alias_to_offset(alias):
        try:
            return pd.tseries.frequencies.to_offset(alias)
        except:
            info = 'invalid offset alias\n\n'
            info += '\tFor more information about offset aliases, see the link below.\n'
            info += '\thttps://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases'
            raise ValueError(info)

    @staticmethod
    def _alias_phrase_to_offset(value):
        """Maps the phrase for an offset alias to an offset object.

        Args:
            value (str): phrase for an offest alias

        Returns:
            BaseOffset: offset object
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
