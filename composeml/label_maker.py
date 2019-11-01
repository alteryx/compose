from sys import stdout

import pandas as pd
from tqdm import tqdm

from composeml.label_times import LabelTimes
from composeml.offsets import to_offset
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


class Context:
    """Metadata for data slice."""

    def __init__(self, gap=None, window=None, slice_number=None, target_entity=None, target_instance=None):
        """Metadata for data slice.

        Args:
            gap (tuple) : Start and stop time for gap.
            window (tuple) : Start and stop time for window.
            slice (int) : Slice number.
            target_entity (int) : Target entity.
            target_instance (int) : Target instance.
        """
        self.gap = gap or (None, None)
        self.window = window or (None, None)
        self.slice_number = slice_number
        self.target_entity = target_entity
        self.target_instance = target_instance


class DataSlice(pd.DataFrame):
    """Data slice for labeling function."""
    _metadata = ['context']

    @property
    def _constructor(self):
        return DataSlice

    def __str__(self):
        """Metadata of data slice."""
        info = {
            'slice_number': self.context.slice_number,
            self.context.target_entity: self.context.target_instance,
            'window': '[{}, {})'.format(*self.context.window),
            'gap': '[{}, {})'.format(*self.context.gap),
        }

        info = pd.Series(info).to_string()
        return info


class LabelMaker:
    """Automatically makes labels for prediction problems."""

    def __init__(self, target_entity, time_index, labeling_function, window_size=None, label_type=None):
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

    def _get_slices(self, group, gap=None, min_data=None, drop_empty=True):
        """Generate data slices for group.

        Args:
            df (DataFrame) : Data frame to generate data slices.
            gap (str or int) : Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            min_data (int or str or Timestamp) : Threshold to cutoff data.
            drop_empty (bool) : Whether to drop empty slices. Default value is True.

        Returns:
            DataSlice : Returns a data slice.
        """
        key, df = group
        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)

        df = df.loc[df.index.notnull()]
        assert df.index.is_monotonic_increasing, "Please sort your dataframe chronologically before calling search"

        if df.empty:
            return

        threshold = min_data or df.index[0]
        df, cutoff_time = cutoff_data(df=df, threshold=threshold)

        if df.empty:
            return

        if isinstance(gap, int):
            cutoff_time = df.index[0]

        df = DataSlice(df)
        df.context = Context(slice_number=0, target_entity=self.target_entity, target_instance=key)

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
                    is_overlap = df_slice.index == window_end

                    if df_slice.index.size > 1 and is_overlap.any():
                        df_slice = df_slice[~is_overlap]

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

    def slice(self, df, num_examples_per_instance, minimum_data=None, gap=None, drop_empty=True, verbose=False):
        """Generates data slices of target entity.

        Args:
            df (DataFrame) : Data frame to create slices on.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            minimum_data (str) : Minimum data before starting search. Default value is first time of index.
            gap (str or int) : Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            drop_empty (bool) : Whether to drop empty slices. Default value is True.
            verbose (bool) : Whether to print metadata about slice. Default value is False.

        Returns:
            DataSlice : Returns data slice.
        """
        if self.window_size is None and gap is None:
            more_than_one = num_examples_per_instance > 1
            assert not more_than_one, "must specify gap if num_examples > 1 and window size = none"

        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)

        df = self.set_index(df)

        if num_examples_per_instance == -1:
            num_examples_per_instance = float('inf')

        for group in df.groupby(self.target_entity):
            slices = self._get_slices(group=group, gap=gap, min_data=minimum_data, drop_empty=drop_empty)

            for df in slices:
                if verbose:
                    print(df)

                yield df

                if df.context.slice_number >= num_examples_per_instance:
                    break

    def search(self,
               df,
               num_examples_per_instance,
               minimum_data=None,
               gap=None,
               drop_empty=True,
               label_type=None,
               verbose=True,
               *args,
               **kwargs):
        """Searches the data to calculates labels.

        Args:
            df (DataFrame) : Data frame to search and extract labels.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            minimum_data (str) : Minimum data before starting search. Default value is first time of index.
            gap (str or int) : Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            drop_empty (bool) : Whether to drop empty slices. Default value is True.
            label_type (str) : The label type can be "continuous" or "categorical". Default value is the inferred label type.
            verbose (bool) : Whether to render progress bar. Default value is True.
            *args : Positional arguments for labeling function.
            **kwargs : Keyword arguments for labeling function.

        Returns:
            LabelTimes : Calculated labels with cutoff times.
        """
        bar_format = "Elapsed: {elapsed} | Remaining: {remaining} | "
        bar_format += "Progress: {l_bar}{bar}| "
        bar_format += self.target_entity + ": {n}/{total} "
        total = len(df.groupby(self.target_entity))
        finite_examples_per_instance = num_examples_per_instance > -1 and num_examples_per_instance != float('inf')

        if finite_examples_per_instance:
            total *= num_examples_per_instance

        progress_bar = tqdm(total=total, bar_format=bar_format, disable=not verbose, file=stdout)

        slices = self.slice(
            df=df,
            num_examples_per_instance=num_examples_per_instance,
            minimum_data=minimum_data,
            gap=gap,
            drop_empty=drop_empty,
            verbose=False,
        )

        name = self.labeling_function.__name__
        labels, instance = [], 0

        for df in slices:
            label = self.labeling_function(df, *args, **kwargs)

            if not pd.isnull(label):
                label = {self.target_entity: df.context.target_instance, 'cutoff_time': df.context.window[0], name: label}
                labels.append(label)

            first_slice_for_instance = df.context.slice_number == 1

            if finite_examples_per_instance:
                progress_bar.update(n=1)

                # update skipped examples for previous instance
                if first_slice_for_instance:
                    instance += 1
                    skipped_examples = instance - 1
                    skipped_examples *= num_examples_per_instance
                    skipped_examples -= progress_bar.n
                    progress_bar.update(n=skipped_examples)

            if not finite_examples_per_instance and first_slice_for_instance:
                progress_bar.update(n=1)

        total -= progress_bar.n
        progress_bar.update(n=total)
        progress_bar.close()

        labels = LabelTimes(data=labels, name=name, target_entity=self.target_entity, label_type=label_type)
        labels = labels.rename_axis('id', axis=0)

        if labels.empty:
            return labels

        if labels.is_discrete:
            labels[labels.name] = labels[labels.name].astype('category')

        labels.name = name
        labels.target_entity = self.target_entity
        labels.settings.update({
            'num_examples_per_instance': num_examples_per_instance,
            'minimum_data': str(minimum_data),
            'window_size': str(self.window_size),
            'gap': str(gap),
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
