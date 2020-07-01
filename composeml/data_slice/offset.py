import re

import pandas as pd


class DataSliceOffset:
    def __init__(self, value):
        self.value = value
        self._check()

    def _check(self):
        if isinstance(self.value, str): self._parse_value()
        assert self._is_valid_offset, self._invalid_offset_error

    @property
    def _is_offset_base(self):
        return issubclass(type(self.value), pd.tseries.offsets.BaseOffset)

    @property
    def _is_offset_position(self):
        return isinstance(self.value, int)

    @property
    def _is_offset_timedelta(self):
        return isinstance(self.value, pd.Timedelta)

    @property
    def _is_offset_timestamp(self):
        return isinstance(self.value, pd.Timestamp)

    @property
    def _is_offset_period(self):
        value = self._is_offset_base
        value |= self._is_offset_timedelta
        return value

    def __int__(self):
        if self._is_offset_position: return self.value
        elif self._is_offset_base: return self.value.n
        elif self._is_offset_timedelta: return self.value.value
        else: raise TypeError('offset must be position or period based')

    def __float__(self):
        if self._is_offset_timestamp: return self.value.timestamp()
        else: raise TypeError('offset must be a timestamp')

    @property
    def _is_positive(self):
        timestamp = self._is_offset_timestamp
        numeric = float if timestamp else int
        return numeric(self) > 0

    @property
    def _is_valid_offset(self):
        value = self._is_offset_position
        value |= self._is_offset_period
        value |= self._is_offset_timestamp
        return value

    @property
    def _invalid_offset_error(self):
        info = 'invalid offset\n\n'
        info += '\tFor information about offset aliases, visit the link below.\n'
        info += '\thttps://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases'
        return info

    def _parse_offset_alias(self, alias):
        value = self._parse_offset_alias_phrase(alias)
        value = value or pd.tseries.frequencies.to_offset(alias)
        return value

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

    def _parse_value(self):
        for parser in self._parsers:
            try:
                value = parser(self.value)
                if value is not None:
                    self.value = value
                    break
            except Exception:
                continue

    @property
    def _parsers(self):
        return pd.Timestamp, self._parse_offset_alias, pd.Timedelta


class DataSliceStep(DataSliceOffset):
    @property
    def _is_valid_offset(self):
        value = self._is_offset_position
        value |= self._is_offset_period
        return value

    @property
    def _parsers(self):
        return self._parse_offset_alias, pd.Timedelta
