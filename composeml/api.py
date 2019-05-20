import pandas as pd


def on_slice(apply, window, min_data, gap, n_examples):
    def df_to_labels(df, *args, **kwargs):
        labels = pd.Series()
        start_time = df.index[0] + min_data

        for example in range(n_examples):
            df = df[str(start_time):]
            df_slice = df[:str(start_time + window)]

            label = apply(df_slice, *args, **kwargs)
            if label is None or label is pd.np.nan:
                continue

            labels[start_time] = label
            start_time += gap

        return labels.rename_axis('time')

    return df_to_labels


class LabelMaker:
    def __init__(self, target_entity, time_index, labeling_function, window_size):
        self.target_entity = target_entity
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = window_size

    def search(self, df, minimum_data, num_examples_per_instance, gap, *args, **kwargs):
        if self.time_index != df.index:
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
        return labels
