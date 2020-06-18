import pandas as pd
from composeml.data_slice import DataSlice, DataSliceContext
from composeml.offset import to_offset
from composeml.utils import can_be_type


class DataSliceGenerator:
    def __init__(self, window_size, gap=None, min_data=None, drop_empty=True):
        self._set_window_size(window_size)
        self.gap = to_offset(gap or self.window_size)
        self.drop_empty = drop_empty
        self.min_data = min_data

    def __call__(self, df):
        data_slices = self._slice_by_time(
            df=df,
            gap=self.gap,
            min_data=self.min_data,
            drop_empty=self.drop_empty,
        )

        for ds in data_slices:
            yield ds

    def _set_window_size(self, window_size):
        """Set and format initial window size parameter.

        Args:
            window_size (str or int): Duration of each data slice.
                The default value for window size is all future data.
        """
        if window_size is not None:
            window_size = to_offset(window_size)

        self.window_size = window_size

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

    def _slice_by_time(self, df, gap=None, min_data=None, drop_empty=True):
        """Generate data slices for a group.

        Args:
            df (DataFrame): Data frame to generate data slices.
            gap (str or int): Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            min_data (int or str or Timestamp): Threshold to cutoff data.
            drop_empty (bool): Whether to drop empty slices. Default value is True.

        Returns:
            df_slice (generator): Returns a generator of data slices.
        """
        df = df.loc[df.index.notnull()]
        assert df.index.is_monotonic_increasing, "Please sort your dataframe chronologically before calling search"

        if df.empty:
            return

        threshold = min_data or df.index[0]
        df, cutoff_time = self._cutoff_data(df=df, threshold=threshold)

        if df.empty:
            return

        if isinstance(gap, int):
            cutoff_time = df.index[0]

        df = DataSlice(df)
        df.context = DataSliceContext(slice_number=0)

        def iloc(index, i):
            if i < index.size:
                return index[i]

        while not df.empty and cutoff_time <= df.index[-1]:
            if isinstance(self.window_size, int):
                df_slice = df.iloc[:self.window_size]
                window_end = iloc(df.index, self.window_size)

            else:
                window_end = cutoff_time + self.window_size
                df_slice = df[:window_end]

                # Pandas includes both endpoints when slicing by time.
                # This results in the right endpoint overlapping in consecutive data slices.
                # Resolved by making the right endpoint exclusive.
                # https://pandas.pydata.org/pandas-docs/version/0.19/gotchas.html#endpoints-are-inclusive

                if not df_slice.empty:
                    overlap = df_slice.index == window_end
                    if overlap.any():
                        df_slice = df_slice[~overlap]

            df_slice.context.window = (cutoff_time, window_end)

            if isinstance(gap, int):
                gap_end = iloc(df.index, gap)
                df_slice.context.gap = (cutoff_time, gap_end)
                df = df.iloc[gap:]

                if not df.empty:
                    cutoff_time = df.index[0]

            else:
                gap_end = cutoff_time + gap
                df_slice.context.gap = (cutoff_time, gap_end)
                cutoff_time += gap

                if cutoff_time <= df.index[-1]:
                    df = df[cutoff_time:]

            if df_slice.empty and drop_empty:
                continue

            df.context.slice_number += 1

            yield df_slice
