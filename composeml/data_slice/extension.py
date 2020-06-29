import pandas as pd
from composeml.utils import can_be_type


class DataSliceContext:
    """Tracks contextual information about the data slices."""
    def __init__(self, start=None, stop=None, step=None, count=0):
        """Creates data slice context.

        Args:
            start: When data slice starts.
            stop: When the data slice stops.
            step: When the next data slice starts.
            count (int): The latest count of data slices.
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.count = 0


class DataSliceFrame(pd.DataFrame):
    """Subclasses pandas data frame for data slice."""
    _metadata = ['context']

    @property
    def _constructor(self):
        return DataSliceFrame

    @property
    def _info(self):
        return vars(self.context)

    def __str__(self):
        info = pd.Series(self._info)
        return info.to_string()


@pd.api.extensions.register_dataframe_accessor("slice")
class DataSliceExtension:
    def __init__(self, df):
        self._df = df

    def __call__(self, size, start=None, step=None, drop_empty=True):
        """Generates data slices from the data frame.

        Args:
            size (str or int): The size of the data slices.
            start (int or str or Timestamp): Where to start the first data slice.
            step (str or int): The step size between data slices. Default value is the data slice size.
            drop_empty (bool): Whether to drop empty data slices. Default value is True.

        Returns:
            df_slice (generator): Returns a generator of data slices.
        """
        window_size, gap, min_data = size, step, start

        df = self._df.loc[df.index.notnull()]
        if df.empty: return

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        threshold = min_data or df.index[0]
        df, cutoff_time = self._cutoff_data(df=df, threshold=threshold)
        if df.empty: return

        if isinstance(gap, int):
            cutoff_time = df.index[0]

        df = DataSliceFrame(df)
        df.context = DataSliceContext()

        def iloc(index, i):
            if i < index.size:
                return index[i]

        while not df.empty and cutoff_time <= df.index[-1]:
            if isinstance(window_size, int):
                df_slice = df.iloc[:window_size]
                window_end = iloc(df.index, window_size)

            else:
                window_end = cutoff_time + window_size
                df_slice = df[:window_end]

                # Pandas includes both endpoints when slicing by time.
                # This results in the right endpoint overlapping in consecutive data slices.
                # Resolved by making the right endpoint exclusive.
                # https://pandas.pydata.org/pandas-docs/version/0.19/gotchas.html#endpoints-are-inclusive

                if not df_slice.empty:
                    overlap = df_slice.index == window_end
                    if overlap.any():
                        df_slice = df_slice[~overlap]

            df_slice.context.start = cutoff_time
            df_slice.context.stop = window_end

            if isinstance(gap, int):
                gap_end = iloc(df.index, gap)
                df_slice.context.step = gap_end
                df = df.iloc[gap:]

                if not df.empty:
                    cutoff_time = df.index[0]

            else:
                gap_end = cutoff_time + gap
                df_slice.context.step = gap_end
                cutoff_time += gap

                if cutoff_time <= df.index[-1]:
                    df = df[cutoff_time:]

            if df_slice.empty and drop_empty: continue
            df.context.count += 1
            yield df_slice

    def _cutoff_data(self, df, threshold):
        """Cuts off data before the threshold.

        Args:
            df (DataFrame): Data frame to cutoff data.
            threshold (int or str or Timestamp): Threshold to apply on data.
                If integer, the threshold will be the time at `n + 1` in the index.
                If string, the threshold can be an offset or timestamp.
                An offset will be applied relative to the first time in the index.

        Returns:
            df, cutoff_time (tuple(DataFrame, Timestamp)): Returns the data frame and the applied cutoff time.
        """
        if isinstance(threshold, int):
            assert threshold > 0, 'threshold must be greater than zero'
            df = df.iloc[threshold:]

            if df.empty:
                return df, None

            cutoff_time = df.index[0]

        elif isinstance(threshold, str):
            if can_be_type(type=pd.tseries.frequencies.to_offset, string=threshold):
                threshold = pd.tseries.frequencies.to_offset(threshold)
                assert threshold.n > 0, 'threshold must be greater than zero'
                cutoff_time = df.index[0] + threshold

            elif can_be_type(type=pd.Timestamp, string=threshold):
                cutoff_time = pd.Timestamp(threshold)

            else:
                raise ValueError('invalid threshold')

        else:
            is_timestamp = isinstance(threshold, pd.Timestamp)
            assert is_timestamp, 'invalid threshold'
            cutoff_time = threshold

        if cutoff_time != df.index[0]:
            df = df[df.index >= cutoff_time]

            if df.empty:
                return df, None

        return df, cutoff_time