import pandas as pd

from composeml.data_slice.offset import DataSliceOffset, DataSliceStep


class DataSliceContext:
    """Tracks contextual attributes about a data slice."""
    def __init__(self, slice_number=0, slice_start=None, slice_stop=None, next_start=None):
        """Creates the data slice context.

        Args:
            slice_number (int): The latest count of data slices.
            slice_start (int or Timestamp): When the data slice starts.
            slice_stop (int or Timestamp): When the data slice stops.
            next_start (int or Timestamp): When the next data slice starts.
        """
        self.next_start = next_start
        self.slice_stop = slice_stop
        self.slice_start = slice_start
        self.slice_number = 0

    def __repr__(self):
        """Represents the data slice context as a strings."""
        return self._series.to_string()

    @property
    def _series(self):
        """Represents the data slice context as a pandas series."""
        keys = reversed(list(vars(self)))
        attrs = {key: getattr(self, key) for key in keys}
        context = pd.Series(attrs, name='context')
        return context

    @property
    def count(self):
        """Alias for the data slice number."""
        return self.slice_number

    @property
    def start(self):
        """Alias for the start point of a data slice."""
        return self.slice_start

    @property
    def stop(self):
        """Alias for the stopping point of a data slice."""
        return self.slice_stop


class DataSliceFrame(pd.DataFrame):
    """Subclasses pandas data frames for data slices."""
    _metadata = ['context']

    @property
    def _constructor(self):
        return DataSliceFrame

    @property
    def ctx(self):
        """Alias for the data slice context."""
        return self.context

    def __str__(self):
        """Returns the data slice context as the printed representation."""
        return repr(self.ctx)


@pd.api.extensions.register_dataframe_accessor("slice")
class DataSliceExtension:
    def __init__(self, df):
        self._df = df

    def __call__(self, size, start=None, step=None, drop_empty=True):
        """Generates data slices from the data frame.

        Args:
            size (int or str): The data size of each data slice. An integer represents the number of rows.
                A string represents a period after the starting point of a data slice.
            start (int or str): Where to start the first data slice.
            step (int or str): The step size between data slices. Default value is the data slice size.
            drop_empty (bool): Whether to drop empty data slices. Default value is True.

        Returns:
            ds (generator): Returns a generator of data slices.
        """
        self._check_index()
        size, start, step = self._check_parameters(size, start, step)
        df = self._apply_start(self._df, start)

        if df.empty: return
        if step._is_offset_position:
            start.value = df.index[0]

        df = DataSliceFrame(df)
        slice_number, stop_value = 1, df.index[-1]
        while not df.empty and start.value <= stop_value:
            ds = self._apply_size(df, start, size)
            df, ds = self._apply_step(df, ds, start, step)
            if ds.empty and drop_empty: continue
            ds.context.slice_number = slice_number
            slice_number += 1
            yield ds

    def _apply_size(self, df, start, size):
        """Returns a data slice calculated by the offsets."""
        if size._is_offset_position:
            ds = df.iloc[:size.value]
            stop = self._iloc(df.index, size.value)
        else:
            stop = start.value + size.value
            ds = df[:stop]

            # Pandas includes both endpoints when slicing by time.
            # This results in the right endpoint overlapping in consecutive data slices.
            # Resolved by making the right endpoint exclusive.
            # https://pandas.pydata.org/pandas-docs/version/0.19/gotchas.html#endpoints-are-inclusive

            if not ds.empty:
                overlap = ds.index == stop
                if overlap.any():
                    ds = ds[~overlap]

        ds.context = DataSliceContext(slice_start=start.value, slice_stop=stop)
        return ds

    def _apply_start(self, df, start):
        """Returns a data frame starting at the offset."""
        if start._is_offset_position and start._is_positive:
            df = df.iloc[start.value:]

            if not df.empty:
                start.value = df.index[0]

        if start._is_offset_period:
            start.value += df.index[0]

        if start._is_offset_timestamp and start.value != df.index[0]:
            df = df[df.index >= start.value]

        return df

    def _apply_step(self, df, ds, start, step):
        """Strides the index starting point by the offset."""
        if step._is_offset_position:
            next_start = self._iloc(df.index, step.value)
            ds.context.next_start = next_start
            df = df.iloc[step.value:]

            if not df.empty:
                start.value = df.index[0]
        else:
            next_start = start.value + step.value
            ds.context.next_start = next_start
            start.value += step.value

            if start.value <= df.index[-1]:
                df = df[start.value:]

        return df, ds

    def _check_parameters(self, size, start, step):
        """Checks if parameters are data slice offsets."""
        if not isinstance(size, DataSliceStep):
            size = DataSliceStep(size)

        start = start or self._df.index[0]
        if not isinstance(start, DataSliceOffset):
            start = DataSliceOffset(start)

        step = step or size
        if not isinstance(step, DataSliceStep):
            step = DataSliceStep(step)

        info = 'offset must be positive'
        assert size._is_positive, info
        assert step._is_positive, info

        if any(offset._is_offset_period for offset in (size, start, step)):
            info = 'offset by time requires a time index'
            assert self._is_time_index, info

        return size, start, step

    def _iloc(self, index, i):
        """Helper function for getting index values."""
        if i < index.size:
            return index[i]

    def _check_index(self):
        """Checks if index values are null or unsorted."""
        info = 'index contains null values'
        assert self._df.index.notnull().all(), info
        info = "data frame must be sorted chronologically"
        assert self._is_sorted, info

    @property
    def _is_sorted(self):
        """Whether index values are sorted."""
        return self._df.index.is_monotonic_increasing

    @property
    def _is_time_index(self):
        """Whether the data frame has a time index type."""
        return pd.api.types.is_datetime64_any_dtype(self._df.index)
