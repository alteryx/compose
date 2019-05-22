import pandas as pd


class LabelTimes(pd.DataFrame):
    """A data frame containing labels made by a label maker."""
    _metadata = ['settings']

    @property
    def _constructor(self):
        return LabelTimes

    def describe(self):
        """Prints out label distribution and the settings used to make the labels."""
        labels = self[self.settings['name']]
        distribution = labels.value_counts()
        print(distribution, end='\n\n')
        print(pd.Series(self.settings), end='\n\n')

    def copy(self):
        """
        Makes a copy of this instance.

        Returns:
            labels (LabelTimes) : Copy of labels.
        """
        labels = super().copy()
        labels.settings = self.settings.copy()
        return labels

    def threshold(self, value, inplace=False):
        """
        Creates binary labels by testing if labels are above threshold.

        Args:
            value (float) : Value of threshold.
            inplace (bool) : Modify labels in place.

        Returns:
            labels (LabelTimes) : Instance of labels.
        """
        labels = self if inplace else self.copy()
        name = labels.settings['name']
        labels[name] = labels[name].gt(value)
        labels.settings.update(threshold=value)
        return labels

    def apply_lead(self, lead, inplace=False):
        """
        Shifts the label times earlier for predicting in advance.

        Args:
            lead (str) : Time to shift earlier.
            inplace (bool) : Modify labels in place.

        Returns:
            labels (LabelTimes) : Instance of labels.
        """
        labels = self if inplace else self.copy()
        labels.settings.update(lead=lead)

        names = labels.index.names
        labels.reset_index(inplace=True)
        labels['time'] = labels['time'].sub(pd.Timedelta(lead))
        labels.set_index(names, inplace=True)

        return labels


def on_slice(make_label, window, min_data, gap, n_examples):
    """
    Returns a function that transforms a data frame to labels.

    Args:
        make_label (function) : Function that transforms a data slice to a label.
        window (Timedelta) : Duration of each data slice.
        min_data (Timedelta) : Minimum data before starting search.
        n_examples (int) : Number of labels to make.
        gap (Timedelta) : Time between examples.

    Returns:
        df_to_labels (function) : Function that transforms a data frame to labels.
    """
    def df_to_labels(df, *args, **kwargs):
        labels = pd.Series()
        cutoff_time = df.index[0] + min_data

        for example in range(n_examples):
            df = df[str(cutoff_time):]
            df_slice = df[:str(cutoff_time + window)]

            label = make_label(df_slice, *args, **kwargs)
            if label is None or label is pd.np.nan:
                continue

            labels[cutoff_time] = label
            cutoff_time += gap

        return labels.rename_axis('time')

    return df_to_labels


class LabelMaker:
    """Automatically makes labels for prediction problems."""
    def __init__(self, target_entity, time_index, labeling_function, window_size):
        """
        Creates an instance of label maker.

        Args:
            target_entity (str) : Entity on which to make labels.
            time_index (str): Name of time column in the data frame.
            labeling_function (function) : Function that transforms a data slice to a label.
            window_size (str) : Duration of each data slice.
        """
        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = window_size

    def search(self, df, minimum_data, num_examples_per_instance, gap, *args, **kwargs):
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
        if df.index.name != self.time_index:
            df = df.set_index(self.time_index)

        df_to_labels = on_slice(
            self.labeling_function,
            min_data=pd.Timedelta(minimum_data),
            window=pd.Timedelta(self.window_size),
            gap=pd.Timedelta(gap),
            n_examples=num_examples_per_instance,
        )

        labels = df.groupby(self.target_entity).apply(df_to_labels, *args, **kwargs)
        labels = labels.to_frame(self.labeling_function.__name__)
        labels = LabelTimes(labels)

        labels.settings = {
            'name': self.labeling_function.__name__,
            'target_entity': self.target_entity,
            'num_examples_per_instance': num_examples_per_instance,
            'minimum_data': minimum_data,
            'window_size': self.window_size,
            'gap': gap,
        }

        return labels
