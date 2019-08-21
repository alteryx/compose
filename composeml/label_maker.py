from sys import stdout

import pandas as pd
from tqdm import tqdm

from composeml.label_times import LabelTimes
from composeml.utils import can_be_type


def cutoff_data(df, threshold):
    """Cuts off data before the threshold.

    Args:
        df (DataFrame) : Data frame to cutoff data.
        threshold (int or str or Timestamp) : Threshold to apply on data.
            If integer, the threshold will be the time at `n + 1` in the index.
            If string, the threshold can be an offset or timestamp.
            An offset will be applied relative to the first time in the index.

    Returns:
        DataFrame, Timestamp : Returns the data frame and the applied cutoff time.
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
        error = 'offset must be a valid string'
        assert can_be_type(type=pd.tseries.frequencies.to_offset, string=value), error
        offset = pd.tseries.frequencies.to_offset(value)
        assert offset.n > 0, 'offset must be greater than zero'

    else:
        assert is_offset(value), 'invalid offset'
        assert value.n > 0, 'offset must be greater than zero'
        offset = value

    return offset


class LabelMaker:
    """Automatically makes labels for prediction problems."""

    def __init__(self, target_entity, time_index, labeling_function, window_size=None):
        """Creates an instance of label maker.

        Args:
            target_entity (str) : Entity on which to make labels.
            time_index (str): Name of time column in the data frame.
            labeling_function (function) : Function that transforms a data slice to a label.
            window_size (str or int) : Duration of each data slice.
                The default value for window size is all future data.
        """
        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = window_size

        if self.window_size is not None:
            self.window_size = to_offset(self.window_size)

    def get_slices(self, df, gap=None, min_data=None, drop_empty=True):
        """Generate data slices.

        Args:
            df (DataFrame) : Data frame to generate data slices.
            gap (str) : Time between slices. Default value is window size.
            min_data (int or str or Timestamp) : Threshold to cutoff data.
            drop_empty (bool) : Whether to drop empty slices.

        Returns:
            DataFrame, dict : Returns a data slice and metadata about the data slice.
        """
        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)

        df = df.loc[df.index.notnull()]
        df.sort_index(inplace=True)

        if df.empty:
            return

        threshold = min_data or df.index[0]
        df, cutoff_time = cutoff_data(df=df, threshold=threshold)

        if df.empty:
            return

        if isinstance(gap, int):
            cutoff_time = df.index[0]

        metadata = {'slice': 0, 'min_data': cutoff_time}

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

                if not df_slice.empty:
                    # exclude last row to avoid overlap
                    is_overlap = df_slice.index == window_end

                    if df_slice.index.size > 1 and is_overlap.any():
                        df_slice = df_slice[~is_overlap]

            metadata['window'] = (cutoff_time, window_end)

            if isinstance(gap, int):
                gap_end = iloc(df.index, gap)
                metadata['gap'] = (cutoff_time, gap_end)
                df = df.iloc[gap:]

                if not df.empty:
                    cutoff_time = df.index[0]

            else:
                gap_end = cutoff_time + gap
                metadata['gap'] = (cutoff_time, gap_end)
                cutoff_time += gap

                if cutoff_time <= df.index[-1]:
                    df = df[cutoff_time:]

            if df_slice.empty and drop_empty:
                continue

            metadata['slice'] += 1

            yield df_slice, metadata

    def slice(self,
              df,
              num_examples_per_instance,
              minimum_data=None,
              gap=None,
              metadata=False,
              drop_empty=True,
              verbose=False):
        """Generates data slices of target entity.

        Args:
            df (DataFrame) : Data frame to create slices on.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            minimum_data (str) : Minimum data before starting search. Default value is first time of index.
            gap (str) : Time between examples. Default value is window size.
            metadata (bool) : Whether to return metadata about the data slice.
            drop_empty (bool) : Whether to drop empty slices.
            verbose (bool) : Whether to print metadata about slice.

        Returns:
            DataFrame : Slice of data.
        """
        if self.window_size is None and gap is None:
            more_than_one = num_examples_per_instance > 1
            assert not more_than_one, "must specify gap if num_examples > 1 and window size = none"

        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)

        df = self.set_index(df)

        if num_examples_per_instance == -1:
            num_examples_per_instance = float('inf')

        for key, df in df.groupby(self.target_entity):
            slices = self.get_slices(df=df, gap=gap, min_data=minimum_data, drop_empty=drop_empty)

            for df_slice, df_metadata in slices:
                df_metadata[self.target_entity] = key

                if verbose:
                    self.print_slice(df_metadata)

                if metadata:
                    df_slice = df_slice, df_metadata

                yield df_slice

                if df_metadata['slice'] >= num_examples_per_instance:
                    break

    def search(self,
               df,
               num_examples_per_instance,
               minimum_data=None,
               gap=None,
               drop_empty=True,
               verbose=True,
               *args,
               **kwargs):
        """Searches the data to calculates labels.

        Args:
            df (DataFrame) : Data frame to search and extract labels.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            minimum_data (str) : Minimum data before starting search.
            gap (str) : Time between examples.
            drop_empty (bool) : Whether to drop empty slices.
            verbose (bool) : Whether to render progress bar.
            *args : Positional arguments for labeling function.
            **kwargs : Keyword arguments for labeling function.

        Returns:
            LabelTimes : Calculated labels with cutoff times.
        """
        if self.window_size is None and gap is None:
            more_than_one = num_examples_per_instance > 1
            assert not more_than_one, "must specify gap if num_examples > 1 and window size = none"

        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)

        bar_format = "Elapsed: {elapsed} | Remaining: {remaining} | "
        bar_format += "Progress: {l_bar}{bar}| "
        bar_format += self.target_entity + ": {n}/{total} "
        total = len(df.groupby(self.target_entity))
        finite = num_examples_per_instance > -1 and num_examples_per_instance != float('inf')

        if finite:
            total *= num_examples_per_instance

        progress_bar = tqdm(total=total, bar_format=bar_format, disable=not verbose, file=stdout)
        name = self.labeling_function.__name__
        labels, instance = [], 0

        slices = self.slice(
            df=df,
            num_examples_per_instance=num_examples_per_instance,
            minimum_data=minimum_data,
            gap=gap,
            metadata=True,
            drop_empty=drop_empty,
            verbose=False)

        for df, metadata in slices:
            label = self.labeling_function(df, *args, **kwargs)

            if not pd.isnull(label):
                key = df[self.target_entity].iloc[0]
                cutoff_time = metadata['window'][0]
                label = {self.target_entity: key, 'cutoff_time': cutoff_time, name: label}
                labels.append(label)

            new_instance = metadata['slice'] == 1

            if finite:
                progress_bar.update(n=1)

                if new_instance:
                    instance += 1
                    n = instance - 1
                    n *= num_examples_per_instance
                    n -= progress_bar.n
                    progress_bar.update(n=n)

            if not finite and new_instance:
                progress_bar.update(n=1)

        total -= progress_bar.n
        progress_bar.update(n=total)
        progress_bar.close()

        labels = LabelTimes(data=labels, name=name, target_entity=self.target_entity)
        labels = labels.rename_axis('id', axis=0)
        labels = labels._with_plots()

        if labels.empty:
            return labels

        labels.settings.update({
            'num_examples_per_instance': num_examples_per_instance,
            'minimum_data': minimum_data,
            'window_size': self.window_size,
            'gap': gap,
        })

        return labels

    def set_index(self, df):
        """Sets the time index in a data frame (if not already set).

        Args:
            df (DataFrame) : Data frame to set time index in.

        Returns:
            DataFrame : Data frame with time index set.
        """
        if df.index.name != self.time_index:
            df = df.set_index(self.time_index)

        if 'time' not in str(df.index.dtype):
            df.index = df.index.astype('datetime64[ns]')

        return df

    def print_slice(self, metadata):
        """Print metadata about slice.

        Args:
            metadata (dict) : metadata about slice
        """
        empty = [None, None]
        window = metadata.get('window', empty)
        gap = metadata.get('gap', empty)

        info = {
            'slice': metadata['slice'],
            self.target_entity: metadata.get(self.target_entity),
            'window': '[{}, {})'.format(*window),
            'gap': '[{}, {})'.format(*gap),
        }

        print(pd.Series(info).to_string(), end='\n\n')
