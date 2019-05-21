import pandas as pd


class LabelTimes(pd.DataFrame):
    _metadata = ['settings']

    @property
    def _constructor(self):
        return LabelTimes

    def describe(self):
        labels = self[self.settings['name']]
        distribution = labels.value_counts()
        print(distribution, end='\n\n')
        print(pd.Series(self.settings), end='\n\n')

    def copy(self):
        label_times = super().copy()
        label_times.settings = self.settings.copy()
        return label_times

    def threshold(self, value, inplace=False):
        label_times = self if inplace else self.copy()
        name = label_times.settings['name']
        label_times[name] = label_times[name].gt(value)
        label_times.settings.update(threshold=value)
        return label_times

    def apply_lead(self, lead, inplace=False):
        label_times = self if inplace else self.copy()
        label_times.settings.update(lead=lead)
        values = label_times.index.get_level_values('time') - pd.Timedelta(lead)
        label_times.index.set_levels(values, level='time', inplace=True)
        return label_times


def on_slice(make_label, window, min_data, gap, n_examples):
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
    def __init__(self, target_entity, time_index, labeling_function, window_size):
        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = window_size

    def search(self, df, minimum_data, num_examples_per_instance, gap, *args, **kwargs):
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
