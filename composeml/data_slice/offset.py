import re

import pandas as pd


class DataSliceOffset:
    """Offsets for calculating data slice indices."""

    def __init__(self, value):
        self.value = value
        self._check()

    def _check(self):
        """Checks if the value is a valid offset."""
        if isinstance(self.value, str):
            self._parse_value()
        assert self._is_valid_offset, self._invalid_offset_error

    @property
    def _is_offset_base(self):
        """Whether offset is a base type."""
        return issubclass(type(self.value), pd.tseries.offsets.BaseOffset)

    @property
    def _is_offset_position(self):
        """Whether offset is integer-location based."""
        return pd.api.types.is_integer(self.value)

    @property
    def _is_offset_timedelta(self):
        """Whether offset is a timedelta."""
        return isinstance(self.value, pd.Timedelta)

    @property
    def _is_offset_timestamp(self):
        """Whether offset is a timestamp."""
        return isinstance(self.value, pd.Timestamp)

    @property
    def _is_offset_frequency(self):
        """Whether offset is a base type or timedelta."""
        value = self._is_offset_base
        value |= self._is_offset_timedelta
        return value

    def __int__(self):
        """Typecasts offset value to an integer."""
        if self._is_offset_position:
            return self.value
        elif self._is_offset_base:
            return self.value.n
        elif self._is_offset_timedelta:
            return self.value.value
        else:
            raise TypeError("offset must be position or frequency based")

    def __float__(self):
        """Typecasts offset value to a float."""
        if self._is_offset_timestamp:
            return self.value.timestamp()
        else:
            raise TypeError("offset must be a timestamp")

    @property
    def _is_positive(self):
        """Whether the offset value is positive."""
        timestamp = self._is_offset_timestamp
        numeric = float if timestamp else int
        return numeric(self) > 0

    @property
    def _is_valid_offset(self):
        """Whether offset is a valid type."""
        value = self._is_offset_position
        value |= self._is_offset_frequency
        value |= self._is_offset_timestamp
        return value

    @property
    def _invalid_offset_error(self):
        """Returns message for invalid offset."""
        info = "offset must be position or time based\n\n"
        info += "\tFor information about offset aliases, visit the link below.\n"
        info += (
            "\thttps://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases"
        )
        return info

    def _parse_offset_alias(self, alias):
        """Parses an alias to an offset."""
        value = self._parse_offset_alias_phrase(alias)
        value = value or pd.tseries.frequencies.to_offset(alias)
        return value

    def _parse_offset_alias_phrase(self, value):
        """Parses an alias phrase to an offset."""
        pattern = re.compile("until start of next (?P<unit>[a-z]+)")
        match = pattern.search(value.lower())

        if match:
            match = match.groupdict()
            unit = match["unit"]

            if unit == "month":
                return pd.offsets.MonthBegin()

            if unit == "year":
                return pd.offsets.YearBegin()

    def _parse_value(self):
        """Parses the value to an offset."""
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
        """Returns the value parsers."""
        return pd.Timestamp, self._parse_offset_alias, pd.Timedelta


class DataSliceStep(DataSliceOffset):
    @property
    def _is_valid_offset(self):
        """Whether offset is a valid type."""
        value = self._is_offset_position
        value |= self._is_offset_frequency
        return value

    @property
    def _parsers(self):
        """Returns the value parsers."""
        return self._parse_offset_alias, pd.Timedelta
