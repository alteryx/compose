from sys import stdout

from tqdm import tqdm

from composeml.data_slice import DataSliceGenerator
from composeml.label_search import ExampleSearch, LabelSearch
from composeml.label_times import LabelTimes


class LabelMaker:
    """Automatically makes labels for prediction problems."""

    def __init__(self, target_entity, time_index, labeling_function=None, window_size=None):
        """Creates an instance of label maker.

        Args:
            target_entity (str): Entity on which to make labels.
            time_index (str): Name of time column in the data frame.
            labeling_function (function or list(function) or dict(str=function)): Function, list of functions, or dictionary of functions that transform a data slice.
                When set as a dictionary, the key is used as the name of the labeling function.
            window_size (str or int): Size of the data slices. As a string, the value can be a timedelta or a column in the data frame to group by.
                As an integer, the value can be the number of rows. Default value is all future data.
        """
        self.labeling_function = labeling_function or {}
        self.target_entity = target_entity
        self.time_index = time_index
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

    def slice(self, df, num_examples_per_instance, minimum_data=None, gap=None, drop_empty=True):
        """Generates data slices of target entity.

        Args:
            df (DataFrame): Data frame to create slices on.
            num_examples_per_instance (int): Number of examples per unique instance of target entity.
            minimum_data (str): Minimum data before starting search. Default value is first time of index.
            gap (str or int): Time between examples. Default value is window size.
                If an integer, search will start on the first event after the minimum data.
            drop_empty (bool): Whether to drop empty slices. Default value is True.

        Returns:
            ds (generator): Returns a generator of data slices.
        """
        self._check_example_count(num_examples_per_instance, gap)
        df = self.set_index(df)
        entity_groups = df.groupby(self.target_entity)
        num_examples_per_instance = ExampleSearch._check_number(num_examples_per_instance)

        generator = DataSliceGenerator(
            window_size=self.window_size,
            min_data=minimum_data,
            drop_empty=drop_empty,
            gap=gap,
        )

        for entity_id, df in entity_groups:
            for ds in generator(df):
                setattr(ds.context, self.target_entity, entity_id)
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

    def _run_search(
        self,
        df,
        generator,
        search,
        verbose=True,
        *args,
        **kwargs,
    ):
        """Search implementation to make label records.

        Args:
            df (DataFrame): Data frame to search and extract labels.
            generator (DataSliceGenerator): The generator for data slices.
            search (LabelSearch or ExampleSearch): The type of search to be done.
            verbose (bool): Whether to render progress bar. Default value is True.
            *args: Positional arguments for labeling function.
            **kwargs: Keyword arguments for labeling function.

        Returns:
            records (list(dict)): Label Records
        """
        df = self.set_index(df)
        entity_groups = df.groupby(self.target_entity)
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

        for entity_count, (entity_id, df) in enumerate(entity_groups):
            for ds in generator(df):
                items = self.labeling_function.items()
                labels = {name: lf(ds, *args, **kwargs) for name, lf in items}
                valid_labels = search.is_valid_labels(labels)
                if not valid_labels: continue

                records.append({
                    self.target_entity: entity_id,
                    'time': ds.context.slice_start,
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
            verbose (bool): Whether to render progress bar. Default value is True.
            *args: Positional arguments for labeling function.
            **kwargs: Keyword arguments for labeling function.

        Returns:
            lt (LabelTimes): Calculated labels with cutoff times.
        """
        assert self.labeling_function, 'missing labeling function(s)'
        self._check_example_count(num_examples_per_instance, gap)
        is_label_search = isinstance(num_examples_per_instance, dict)
        search = (LabelSearch if is_label_search else ExampleSearch)(num_examples_per_instance)

        generator = DataSliceGenerator(
            window_size=self.window_size,
            min_data=minimum_data,
            drop_empty=drop_empty,
            gap=gap,
        )

        records = self._run_search(
            df=df,
            generator=generator,
            search=search,
            verbose=verbose,
            *args,
            **kwargs,
        )

        lt = LabelTimes(
            data=records,
            target_columns=list(self.labeling_function),
            target_entity=self.target_entity,
            search_settings={
                'num_examples_per_instance': num_examples_per_instance,
                'minimum_data': str(minimum_data),
                'window_size': str(self.window_size),
                'gap': str(gap),
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
