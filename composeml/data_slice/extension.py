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
        """Represents the data slice context as a string."""
        return self._series.fillna('').to_string()

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
    """Subclasses pandas data frame for data slice."""
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

    def __call__(self, size=None, start=None, stop=None, step=None, drop_empty=True):
        """Generates data slices from the data frame.

        Args:
            size (int or str): The data size of each data slice. An integer represents the number of rows.
                A string represents a period after the starting point of a data slice.
            start (int or str): Where to start the first data slice.
            stop (int or str): Where to stop generating data slices.
            step (int or str): The step size between data slices. Default value is the data slice size.
            drop_empty (bool): Whether to drop empty data slices. Default value is True.

        Returns:
            ds (generator): Returns a generator of data slices.
        """
        self._check_index()
        size, start, stop, step = self._check_parameters(size, start, stop, step)
        df = self._apply_start(self._df, start)
        if not df.empty: self._apply_stop(df, stop)
        else: return df

        if step._is_offset_position:
            start.value = df.index[0]

        df, slice_number = DataSliceFrame(df), 1
        while not df.empty and start.value <= stop.value:
            ds = self._apply_size(df, start, size)
            df = self._apply_step(df, start, step)
            if ds.empty and drop_empty: continue
            ds.context.next_start = start.value
            ds.context.slice_number = slice_number
            slice_number += 1
            yield ds

    def __getitem__(self, offset):
        """Generates data slices from a slice object."""
        if not isinstance(offset, slice): raise TypeError('must be a slice object')
        return self(size=offset.step, start=offset.start, stop=offset.stop)

    def _apply_size(self, df, start, size):
        """Returns a data slice calculated by the offsets."""
        if size._is_offset_position:
            ds = df.iloc[:size.value]
            stop = self._iloc(df.index, size.value) or df.index[-1]
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
        """Removes data before the index value calculated by the offset."""
        if start._is_offset_period:
            start.value += df.index[0]

        inplace = start.value == df.index[0]
        if start._is_offset_position and not inplace:
            df = df.iloc[start.value:]
            if not df.empty: 
                start.value = df.index[0] 

        if start._is_offset_timestamp and not inplace:
            df = df[df.index >= start.value]

        return df

    def _apply_stop(self, df, stop):
        """Removes data after the index value calculated by the offset."""
        if stop._is_offset_period:
            stop.value += df.index[-1]

        inplace = stop.value == df.index[-1]
        if stop._is_offset_position and not inplace:
            stop.value = self._iloc(df.index, stop.value - 1) or df.index[-1]

    def _apply_step(self, df, start, step):
        """Strides the index starting point by the offset."""
        if step._is_offset_position:
            df = df.iloc[step.value:]
            start.value = self._iloc(df.index, 0)

        else:
            start.value += step.value
            if start.value <= df.index[-1]:
                df = df[start.value:]

        return df

    def _check_parameters(self, size, start, stop, step):
        """Checks if parameters are data slice offsets."""
        size = size or len(self._df)
        if not isinstance(size, DataSliceStep):
            size = DataSliceStep(size)

        start = start or self._df.index[0]
        if not isinstance(start, DataSliceOffset):
            start = DataSliceOffset(start)

        stop = stop or self._df.index[-1]
        if not isinstance(stop, DataSliceOffset):
            stop = DataSliceOffset(stop)

        step = step or size
        if not isinstance(step, DataSliceStep):
            step = DataSliceStep(step)

        info = 'offset must be positive'
        assert size._is_positive, info
        assert step._is_positive, info

        offsets = size, start, stop, step
        if any(offset._is_offset_period for offset in offsets):
            info = 'offset by time requires a time index'
            assert self._is_time_index, info

        return size, start, stop, step

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
