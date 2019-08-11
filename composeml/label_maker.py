from sys import stdout

import pandas as pd
from tqdm import tqdm

from composeml.label_times import LabelTimes


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
        # assert_valid_offset(window_size)
        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = self._process_offset(window_size, name='window size')
        self.gap_size = None

    def _process_df(self, df):
        if df.index.name != self.time_index:
            df = df.set_index(self.time_index)

        if 'time' not in str(df.index.dtype):
            df.index = df.index.astype('datetime64[ns]')

        return df

    def _process_min_data(self, min_data, index):
        if min_data is None:
            min_data = index[0]

        if isinstance(min_data, int):
            error = 'minimum data must be greater than zero'
            assert min_data > 0, error
            index = index[min_data:]

            if index.empty:
                return None, index

            min_data = index[0]

        if isinstance(min_data, str):
            min_data = pd.tseries.frequencies.to_offset(min_data)
            error = 'minimum data must be greater than zero'
            assert min_data.n > 0, error
            min_data += index[0]

        if isinstance(min_data, pd.Timestamp) and min_data != index[0]:
            index = index[index >= min_data]

            if index.empty:
                return None, index

        error = 'minimum data must be an integer, string, or timestamp'
        assert isinstance(min_data, pd.Timestamp), error
        return min_data, index

    def _process_offset(self, offset, name):
        if isinstance(offset, int):
            error = f'{name} must be greater than zero'
            assert offset > 0, error

        elif isinstance(offset, str):
            offset = pd.tseries.frequencies.to_offset(offset)
            error = f'{name} must be greater than zero'
            assert offset.n > 0, error

        else:
            if not issubclass(offset, pd.tseries.offsets.BaseOffset):
                raise ValueError(f'invalid {name}')

        return offset

    def _process_slice(self, df, cutoff_time, gap_size, gap_end):
        df = df[cutoff_time:]

        if gap_size == self.window_size:
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
        if num_examples_per_instance == -1:
            num_examples_per_instance = float('inf')

        if gap is None:
            gap = self.window_size

        self.gap_size = self._process_offset(gap, name='gap')

        df = self._process_df(df)

        for key, df in df.groupby(self.target_entity):
            df = df.loc[df.index.notnull()]
            df.sort_index(inplace=True)

            if df.index.empty:
                continue

            cutoff_time, index = self._process_min_data(min_data=minimum_data, index=df.index)

            if index.empty:
                continue

            if isinstance(self.gap_size, int):
                intervals = iterate_by_range(index=index, offset=self.gap_size)
            else:
                intervals = iterate_by_time(index=index, offset=self.gap_size, start=cutoff_time)

            n_examples = 0

            for cutoff_time, gap_end in intervals:
                df_slice, window_end = self._process_slice(
                    df=df,
                    cutoff_time=cutoff_time,
                    gap_size=self.gap_size,
                    gap_end=gap_end,
                )

                if df_slice.empty:
                    continue

                n_examples += 1

                if verbose:
                    info = pd.Series()
                    info[self.target_entity] = key
                    info['slice'] = n_examples
                    info['window'] = '[{}, {})'.format(cutoff_time, window_end)
                    info['gap'] = '[{}, {})'.format(cutoff_time, gap_end)
                    print(info.to_string(), end='\n\n')

                if edges:
                    df_slice = df_slice, (cutoff_time, window_end)

                yield df_slice

                if n_examples >= num_examples_per_instance:
                    break

    def search(self, df, num_examples_per_instance, minimum_data=None, gap=None, verbose=True, *args, **kwargs):
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

        progress_bar.update(n=num_examples_per_instance - progress_bar.n)
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
