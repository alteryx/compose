from sys import stdout

import pandas as pd
from tqdm import tqdm

from composeml.label_times import LabelTimes


def assert_valid_offset(value):
    if isinstance(value, int):
        assert value >= 0, 'negative offset'

    elif isinstance(value, str):
        offset = pd.Timedelta(value)
        assert offset is not pd.NaT, 'invalid offset'
        assert offset.total_seconds() >= 0, 'negative offset'

    else:
        raise TypeError('invalid offset type')


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

        fast_forward = interval * int(elapsed / interval)
        fast_forward = pd.Timedelta(f'{fast_forward}s')

        yield start, start + offset
        start += fast_forward


def iterate_by_range(index, offset):
    for i in index:
        if i % offset == 0:
            yield i, i + offset


def offset_time(index, value):
    if isinstance(value, int):
        value += 1
        value = index[:value][-1]
        return value

    if isinstance(value, str):
        value = pd.Timedelta(value)
        value = index[0] + value
        return value


class LabelMaker:
    """Automatically makes labels for prediction problems."""

    def __init__(self, target_entity, time_index, labeling_function, window_size):
        """
        Creates an instance of label maker.

        Args:
            target_entity (str) : Entity on which to make labels.
            time_index (str): Name of time column in the data frame.
            labeling_function (function) : Function that transforms a data slice to a label.
            window_size (str or int) : Duration of each data slice.
        """
        assert_valid_offset(window_size)

        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = window_size

    def _preprocess_(self, df):
        if df.index.name != self.time_index:
            df = df.set_index(self.time_index)

        if 'time' not in str(df.index.dtype):
            df.index = df.index.astype('datetime64[ns]')

        return df

    def slice(self, df, minimum_data=None, gap=1, edges=False):
        assert_valid_offset(minimum_data)
        assert_valid_offset(gap)

        df = self._preprocess_(df)
        groups = df.groupby(self.target_entity)

        for key, df in groups:
            df = df.loc[df.index.notnull()]
            df.sort_index(inplace=True)
            index = df.index

            if isinstance(minimum_data, str):
                start = pd.tseries.frequencies.to_offset(minimum_data)
                start += index[0]

                index = index[index >= start]

            if isinstance(minimum_data, int):
                index = index[minimum_data:]

            if index.empty:
                continue

            if isinstance(gap, int):
                window = iterate_by_range(index, offset=gap)

            else:
                gap = pd.tseries.frequencies.to_offset(gap)
                window = iterate_by_time(index, offset=gap)

            for start, stop in window:
                window = df[start:stop]

                if window.index[-1] == stop:
                    window = window[:-1]

                if edges:
                    window = window, (start, stop)

                yield window

    def search(self, df, minimum_data, num_examples_per_instance, gap, verbose=True, *args, **kwargs):
        """
        Searches and extracts labels from a data frame.

        Args:
            df (DataFrame) : Data frame to search and extract labels.
            minimum_data (str) : Minimum data before starting search.
            num_examples_per_instance (int) : Number of examples per unique instance of target entity.
            gap (str) : Time between examples.
            args : Positional arguments for labeling function.
            kwargs : Keyword arguments for labeling function.

        Returns:
            labels (LabelTimes) : A data frame of the extracted labels.
        """
        assert_valid_offset(minimum_data)
        assert_valid_offset(gap)

        df = self._preprocess_(df)
        groups = df.groupby(self.target_entity)
        name = self.labeling_function.__name__

        bar_format = "Elapsed: {elapsed} | Remaining: {remaining} | "
        bar_format += "Progress: {l_bar}{bar}| "
        bar_format += self.target_entity + ": {n}/{total} "
        total = groups.ngroups * num_examples_per_instance
        progress_bar = tqdm(total=total, bar_format=bar_format, disable=not verbose, file=stdout)

        def df_to_labels(df):
            labels = pd.Series()
            df = df.loc[df.index.notnull()]
            df.sort_index(inplace=True)

            if df.empty:
                return labels

            cutoff_time = offset_time(df.index, minimum_data)

            for example in range(num_examples_per_instance):
                df = df[cutoff_time:]

                if df.empty:
                    skipped_iterations = num_examples_per_instance - example
                    progress_bar.update(n=skipped_iterations)
                    break

                window_end = offset_time(df.index, self.window_size)
                label = self.labeling_function(df[:window_end], *args, **kwargs)

                if not pd.isnull(label):
                    labels[cutoff_time] = label

                cutoff_time = offset_time(df.index, gap)
                progress_bar.update(n=1)

            labels.index = labels.index.rename('cutoff_time')
            labels.index = labels.index.astype('datetime64[ns]')
            return labels

        labels_per_group = []
        for key, df in groups:
            labels = df_to_labels(df)
            labels = labels.to_frame(name=name)
            labels[self.target_entity] = key
            labels_per_group.append(labels)

        progress_bar.close()
        labels = pd.concat(labels_per_group, axis=0, sort=False)

        if labels.empty:
            return LabelTimes(name=name, target_entity=self.target_entity)

        labels.reset_index(inplace=True)
        labels.rename_axis('label_id', inplace=True)

        sort = [self.target_entity, 'cutoff_time', name]
        labels = LabelTimes(labels[sort], name=name, target_entity=self.target_entity)
        labels = labels._with_plots()

        labels.settings.update({
            'num_examples_per_instance': num_examples_per_instance,
            'minimum_data': minimum_data,
            'window_size': self.window_size,
            'gap': gap,
        })

        return labels
