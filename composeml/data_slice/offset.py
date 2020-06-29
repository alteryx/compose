import re

import pandas as pd


class DataSliceOffset:
    def __init__(self, value):
        self.value = value
        self._check()

    def _check(self):
        if isinstance(self.value, str):
            self.value = self._parse_value()

        assert self._is_valid_offset, self._invalid_offset_error

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

    @property
    def _invalid_offset_error(self):
        info = 'invalid offset\n\n'
        info += '\tFor information about offset aliases, visit the link below.\n'
        info += '\thttps://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases'
        return info

    def _parse_offset_alias(self, alias):
        try:
            value = self._parse_offset_alias_phrase(alias)
            value = value or pd.tseries.frequencies.to_offset(alias)
            return value
        except:
            return

    def _parse_offset_alias_phrase(self, value):
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

    def _parse_timedelta(self, value):
        try:
            return pd.Timedelta(value)
        except:
            return

    def _parse_value(self):
        for parser in self._parsers:
            value = parser(self.value)
            if value: return value

    @property
    def _parsers(self):
        return [self._parse_offset_alias, self._parse_timedelta]


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
