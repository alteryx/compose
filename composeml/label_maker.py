from sys import stdout

import pandas as pd
from tqdm import tqdm

from composeml.data_slice import DataSlice, DataSliceContext
from composeml.label_search import ExampleSearch, LabelSearch
from composeml.label_times import LabelTimes
from composeml.offsets import to_offset
from composeml.utils import can_be_type


def cutoff_data(df, threshold):
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


class LabelMaker:
    """Automatically makes labels for prediction problems."""

    def __init__(self, target_entity, time_index, labeling_function=None, window_size=None, label_type=None):
        """Creates an instance of label maker.

        Args:
            target_entity (str): Entity on which to make labels.
            time_index (str): Name of time column in the data frame.
            labeling_function (function or list(function) or dict(str=function)): Function, list of functions, or dictionary of functions that transform a data slice.
                When set as a dictionary, the key is used as the name of the labeling function.
            window_size (str or int): Duration of each data slice.
                The default value for window size is all future data.
        """
        self._set_window_size(window_size)
        self.labeling_function = labeling_function
        self.target_entity = target_entity
        self.time_index = time_index

    def _set_window_size(self, window_size):
        """Set and format initial window size parameter.

        Args:
            window_size (str or int): Duration of each data slice.
                The default value for window size is all future data.
        """
        if window_size is not None:
            window_size = to_offset(window_size)

        self.window_size = window_size

    def _name_labeling_function(self, function):
        """Gets the names of the labeling functions."""
        has_name = hasattr(function, '__name__')
        return function.__name__ if has_name else type(function).__name__

    def _check_labeling_function(self, function, name=None):
        """Checks whether the labeling function is callable."""
        assert callable(function), 'labeling function must be callabe'
        return function

    @property
    def labeling_function(self):
        """Gets the labeling function(s)."""
        return self._labeling_function

    @labeling_function.setter
    def labeling_function(self, value):
        """Sets and formats the intial labeling function(s).

        Args:
            value (function or list(function) or dict(str=function)): Function that transforms a data slice to a label.
        """
        if isinstance(value, dict):
            for name, function in value.items():
                self._check_labeling_function(function)
                assert isinstance(name, str), 'labeling function name must be string'

        if callable(value):
            value = [value]

        if isinstance(value, (tuple, list)):
            value = {self._name_labeling_function(function): self._check_labeling_function(function) for function in value}

        assert isinstance(value, dict), 'value type for labeling function not supported'
        self._labeling_function = value

    def _slice(self, df, gap=None, min_data=None, drop_empty=True):
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
        df, cutoff_time = cutoff_data(df=df, threshold=threshold)

        if df.empty:
            return

        if isinstance(gap, int):
            cutoff_time = df.index[0]

        df = DataSlice(df)
        df.context = DataSliceContext(slice_number=0, target_entity=self.target_entity)

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

    def slice(self, df, num_examples_per_instance, minimum_data=None, gap=None, drop_empty=True, verbose=False):
        """Generates data slices of target entity.

        Args:
            df (DataFrame): Data frame to create slices on.
            num_examples_per_instance (int): Number of examples per unique instance of target entity.
            minimum_data (str): Minimum data before starting search. Default value is first time of index.
            gap (str or int): Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            drop_empty (bool): Whether to drop empty slices. Default value is True.
            verbose (bool): Whether to print metadata about slice. Default value is False.

        Returns:
            ds (generator): Returns a generator of data slices.
        """
        self._check_example_count(num_examples_per_instance, gap)
        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)
        groups = self.set_index(df).groupby(self.target_entity)

        if num_examples_per_instance == -1:
            num_examples_per_instance = float('inf')

        for key, df in groups:
            slices = self._slice(df=df, gap=gap, min_data=minimum_data, drop_empty=drop_empty)

            for ds in slices:
                ds.context.target_instance = key
                if verbose: print(ds)
                yield ds

                if ds.context.slice_number >= num_examples_per_instance:
                    break

    @property
    def _bar_format(self):
        """Template to format the progress bar during a label search."""
        value = "Elapsed: {elapsed} | "
        value += "Remaining: {remaining} | "
        value += "Progress: {l_bar}{bar}| "
        value += self.target_entity + ": {n}/{total} "
        return value

    def _run_search(self, df, search, gap=None, min_data=None, drop_empty=True, verbose=True, *args, **kwargs):
        """Search implementation to make label records.

        Args:
            df (DataFrame): Data frame to search and extract labels.
            search (LabelSearch or ExampleSearch): The type of search to be done.
            min_data (str): Minimum data before starting search. Default value is first time of index.
            gap (str or int): Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            drop_empty (bool): Whether to drop empty slices. Default value is True.
            verbose (bool): Whether to render progress bar. Default value is True.
            *args: Positional arguments for labeling function.
            **kwargs: Keyword arguments for labeling function.

        Returns:
            records (list(dict)): Label Records
        """
        entity_groups = self.set_index(df).groupby(self.target_entity)
        multiplier = search.expected_count if search.is_finite else 1
        total = entity_groups.ngroups * multiplier

        progress_bar, records = tqdm(
            total=total,
            bar_format=self._bar_format,
            disable=not verbose,
            file=stdout,
        ), []

        def missing_examples(entity_count):
            return entity_count * search.expected_count - progress_bar.n

        for entity_count, group in enumerate(entity_groups):
            entity_id, df = group

            slices = self._slice(
                df=df,
                gap=gap,
                min_data=min_data,
                drop_empty=drop_empty,
            )

            for ds in slices:
                items = self.labeling_function.items()
                labels = {name: lf(ds, *args, **kwargs) for name, lf in items}
                valid_labels = search.is_valid_labels(labels)
                if not valid_labels: continue

                records.append({
                    self.target_entity: entity_id,
                    'time': ds.context.window[0],
                    **labels,
                })

                search.update_count(labels)
                # if finite search, progress bar is updated for each example found
                if search.is_finite: progress_bar.update(n=1)
                if search.is_complete: break

            # if finite search, progress bar is updated for examples not found
            # otherwise, progress bar is updated for each entity group
            n = missing_examples(entity_count + 1) if search.is_finite else 1
            progress_bar.update(n=n)
            search.reset_count()

        total -= progress_bar.n
        progress_bar.update(n=total)
        progress_bar.close()
        return records

    def _records_to_label_times(self, records, label_name, label_type, settings):
        """Makes a label times object from label records.

        Args:
            records (list(dict)): The label records as a result from a label search.
            label_name (str): The column name that contains the label values.
            label_type (str): The type of label values -- must be "continuous" or "discrete".
            settings (dict): The parameter settings used to make the labels.

        Returns:
            lt (LabelTimes): A label times object of the search records.
        """
        lt = LabelTimes(
            data=records,
            name=label_name,
            label_type=label_type,
            target_entity=self.target_entity,
        )

        lt = lt.rename_axis('id', axis=0)
        if lt.empty: return lt
        lt.settings.update(settings)
        return lt

    def _check_example_count(self, num_examples_per_instance, gap):
        """Checks whether example count corresponds to data slices."""
        if self.window_size is None and gap is None:
            more_than_one = num_examples_per_instance > 1
            assert not more_than_one, "must specify gap if num_examples > 1 and window size = none"

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
            df (DataFrame): Data frame to search and extract labels.
            num_examples_per_instance (int or dict): The expected number of examples to return from each entity group.
                A dictionary can be used to further specify the expected number of examples to return from each label.
            minimum_data (str): Minimum data before starting search. Default value is first time of index.
            gap (str or int): Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            drop_empty (bool): Whether to drop empty slices. Default value is True.
            label_type (str): The label type can be "continuous" or "categorical". Default value is the inferred label type.
            verbose (bool): Whether to render progress bar. Default value is True.
            *args: Positional arguments for labeling function.
            **kwargs: Keyword arguments for labeling function.

        Returns:
            lt (LabelTimes): Calculated labels with cutoff times.
        """
        assert self.labeling_function, 'missing labeling function(s)'
        self._check_example_count(num_examples_per_instance, gap)
        self.window_size = self.window_size or len(df)
        gap = to_offset(gap or self.window_size)

        is_label_search = isinstance(num_examples_per_instance, dict)
        search = (LabelSearch if is_label_search else ExampleSearch)(num_examples_per_instance)

        records = self._run_search(
            df=df,
            search=search,
            gap=gap,
            min_data=minimum_data,
            drop_empty=drop_empty,
            verbose=verbose,
            *args,
            **kwargs,
        )

        lt = self._records_to_label_times(
            records=records,
            label_name=list(self.labeling_function)[0],
            label_type=label_type,
            settings={
                'num_examples_per_instance': num_examples_per_instance,
                'minimum_data': str(minimum_data),
                'window_size': str(self.window_size),
                'gap': str(gap)
            },
        )

        return lt

    def set_index(self, df):
        """Sets the time index in a data frame (if not already set).

        Args:
            df (DataFrame): Data frame to set time index in.

        Returns:
            df (DataFrame): Data frame with time index set.
        """
        if df.index.name != self.time_index:
            df = df.set_index(self.time_index)

        if 'time' not in str(df.index.dtype):
            df.index = df.index.astype('datetime64[ns]')

        return df
