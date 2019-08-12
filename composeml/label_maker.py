from sys import stdout

import pandas as pd
from tqdm import tqdm

from composeml.label_times import LabelTimes
from composeml.utils import is_type


def iterate_by_range(index, offset):
    for i in range(index.size):
        if i % offset == 0:
            j = i + offset

            if j >= index.size:
                yield index[i], None
                break

            yield index[i], index[j]


def iterate_by_time(index, offset, start=None):
    if start is None:
        start = index[0]

    for time in index:
        elapsed = time - start
        elapsed = elapsed.total_seconds()

        interval = (start + offset) - start
        interval = interval.total_seconds()

        if elapsed < interval:
            continue

        yield start, start + offset

        fast_forward = interval * int(elapsed / interval)
        fast_forward = pd.Timedelta(f'{fast_forward}s')
        start += fast_forward

    if start + offset > index[-1]:
        yield start, start + offset


def cutoff_data(df, threshold):
    if isinstance(threshold, int):
        assert threshold > 0, 'threshold must be greater than zero'
        df = df.loc[df.index[threshold:]]

        if df.empty:
            return df, None

        cutoff_time = df.index[0]

    elif isinstance(threshold, str):
        if is_type(type=pd.tseries.frequencies.to_offset, string=threshold):
            threshold = pd.tseries.frequencies.to_offset(threshold)
            assert threshold.n > 0, 'threshold must be greater than zero'
            cutoff_time = df.index[0] + threshold

        elif is_type(type=pd.Timestamp, string=threshold):
            cutoff_time = pd.Timestamp(threshold)

        else:
            raise ValueError('invalid threshold')

    else:
        cutoff_time = threshold

    is_timestamp = isinstance(cutoff_time, pd.Timestamp)
    assert is_timestamp, 'invalid threshold'

    if cutoff_time != df.index[0]:
        df = df[df.index >= cutoff_time]

        if df.empty:
            return df, None

    return df, cutoff_time


def to_offset(value):
    if isinstance(value, int):
        assert value > 0, 'offset must be greater than zero'
        offset = value

    elif isinstance(value, str):
        error = 'offset must be a valid string'
        assert is_type(type=pd.tseries.frequencies.to_offset, string=value), error
        offset = pd.tseries.frequencies.to_offset(value)
        assert offset.n > 0, 'offset must be greater than zero'

    else:
        raise ValueError('invalid offset')

    return offset


class LabelMaker:
    """Automatically makes labels for prediction problems."""

    def __init__(self, target_entity, time_index, labeling_function, window_size):
        """Creates an instance of label maker.

        Args:
            target_entity (str) : Entity on which to make labels.
            time_index (str): Name of time column in the data frame.
            labeling_function (function) : Function that transforms a data slice to a label.
            window_size (str or int) : Duration of each data slice.
        """
        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = to_offset(window_size)
        self.gap_size = None

    def set_index(self, df):
        if df.index.name != self.time_index:
            df = df.set_index(self.time_index)

        if 'time' not in str(df.index.dtype):
            df.index = df.index.astype('datetime64[ns]')

        return df

    def get_intervals(self, df, min_data):
        df = df.loc[df.index.notnull()]
        df.sort_index(inplace=True)

        if df.empty:
            return df, None

        if min_data is None:
            min_data = df.index[0]

        df, cutoff_time = cutoff_data(df=df, threshold=min_data)

        if df.empty:
            return df, None

        if isinstance(self.gap_size, int):
            intervals = iterate_by_range(index=df.index, offset=self.gap_size)
        else:
            intervals = iterate_by_time(index=df.index, offset=self.gap_size, start=cutoff_time)

        return df, intervals

    def get_slice(self, df, cutoff_time, gap_end):
        df = df[cutoff_time:]

        if self.gap_size == self.window_size:
            window_end = gap_end
        else:
            if isinstance(self.window_size, int):
                if self.window_size >= df.index.size:
                    window_end = None
                else:
                    window_end = df.index[self.window_size]
            else:
                window_end = cutoff_time + self.window_size

        df_slice = df[:window_end]

        if df_slice.empty:
            return df_slice, window_end

        # exclude last row to avoid overlap
        is_overlap = df_slice.index[-1] == window_end

        if df_slice.size > 1 and is_overlap:
            df_slice = df_slice[:-1]

        return df_slice, window_end

    def slice(self, df, num_examples_per_instance, minimum_data=None, gap=None, edges=False, verbose=False):
        """Generates data slices.

        Args:
            df (DataFrame) : Data frame to create slices on.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            minimum_data (str) : Minimum data before starting search. Default value is first time of index.
            gap (str) : Time between examples. Default value is window size.
            edges (bool) : Whether to return the start time (cutoff time) and stop time of the window.

        Returns:
            DataFrame : Slice of data.
        """
        if num_examples_per_instance == -1:
            num_examples_per_instance = float('inf')

        self.gap_size = to_offset(gap or self.window_size)
        df = self.set_index(df)

        for key, df in df.groupby(self.target_entity):
            df, intervals = self.get_intervals(df=df, min_data=minimum_data)

            if df.empty:
                continue

            n_examples = 0

            for cutoff_time, gap_end in intervals:
                df_slice, window_end = self.get_slice(df=df, cutoff_time=cutoff_time, gap_end=gap_end)

                if df_slice.empty:
                    continue

                n_examples += 1

                if verbose:
                    info = {
                        self.target_entity: key,
                        'slice': n_examples,
                        'window': f'[{cutoff_time}, {window_end})',
                        'gap': f'[{cutoff_time}, {gap_end})',
                    }
                    info = pd.Series(info).to_string()
                    print(info, end='\n\n')

                if edges:
                    df_slice = df_slice, (cutoff_time, window_end)

                yield df_slice

                if n_examples >= num_examples_per_instance:
                    break

    def search(self, df, num_examples_per_instance, minimum_data=None, gap=None, verbose=True, *args, **kwargs):
        """Searches and extracts labels from data.

        Args:
            df (DataFrame) : Data frame to search and extract labels.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            minimum_data (str) : Minimum data before starting search.
            gap (str) : Time between examples.
            verbose (bool) : Whether to render progress bar.
            *args : Positional arguments for labeling function.
            **kwargs : Keyword arguments for labeling function.

        Returns:
            labels (LabelTimes) : A data frame of the extracted labels.
        """
        bar_format = "Elapsed: {elapsed} | Remaining: {remaining} | "
        bar_format += "Progress: {l_bar}{bar}| "
        bar_format += self.target_entity + ": {n}/{total} "

        if num_examples_per_instance == -1 or num_examples_per_instance == float('inf'):
            total = None
            disable = True
        else:
            n_groups = df.groupby(self.target_entity).ngroups
            total = n_groups * num_examples_per_instance
            disable = not verbose

        progress_bar = tqdm(total=total, bar_format=bar_format, disable=disable, file=stdout)
        name = self.labeling_function.__name__
        labels = []

        slices = self.slice(
            df=df,
            num_examples_per_instance=num_examples_per_instance,
            minimum_data=minimum_data,
            gap=gap,
            edges=True,
            verbose=False,
        )

        for df, edges in slices:
            cutoff_time, window_end = edges
            label = self.labeling_function(df, *args, **kwargs)

            if not pd.isnull(label):
                key = df[self.target_entity].iloc[0]
                label = {self.target_entity: key, 'cutoff_time': cutoff_time, name: label}
                labels.append(label)

            progress_bar.update(n=1)

        n = num_examples_per_instance - progress_bar.n
        progress_bar.update(n=n)
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
            'gap': self.gap_size,
        })

        return labels
