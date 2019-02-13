class LabelByInstance(object):
    def __init__(self, instance_id, labeling_function, drop_null_labels, verbose=False):
        self.instance_id = instance_id
        self.labeling_function = labeling_function
        self.drop_null_labels = drop_null_labels
        self.verbose = verbose

    def search(self, dataframe, **kwargs):
        label_times = dataframe.groupby(self.instance_id).apply(self.labeling_function, **kwargs)
        return label_times



class LabelByWindow(object):
    def __init__(self, instance_id, time_index, labeling_function, window_size,
                 uses_cutoff_time, drop_null_labels, verbose=False):

        self.instance_id = instance_id
        self.time_index = time_index
        self.labeling_function = labeling_function
        self.window_size = window_size
        self.uses_cutoff_time = uses_cutoff_time
        self.drop_null_labels = drop_null_labels
        self.verbose = verbose


    def search(self, dataframe, search_params, **kwargs):
        def label_by_instance(instance_df, **kwargs):
            label_times = []
            instance_id = instance_df.iloc[self.instance_id, 0]
            windows = make_windows(instance_df, self.time_index, search_params)
            for cutoff_time, window_df in windows:
                if uses_cutoff_time:
                    kwargs.update({'cutoff_time': cutoff_time})

                label = self.labeling_function(window, **kwargs)
                label_times.append([instance_id, cutoff_time, label])
            return label_times

        lbl = LabelByInstance(self.instance_id, label_by_instance, drop_null_labels=drop_null_labels)
        label_times = lbl.search(dataframe, **kwargs)
        return label_times


class LabelFromEvents(object):

    def __init__(self, instance_id, time_index, window_size, verbose=False):
        self.instance_id = instance_id
        self.time_index = time_index
        self.window_size = window_size
        self.verbose = verbose

    def search(self, dataframe, search_params, **kwargs):
        def window_function(window_df):
            if window_df.shape[0] > 0:
                return True
            return False


        lbl = LabelByWindow(self.instance_id, window_function, drop_null_labels=drop_null_labels)
        label_times = lbl.search(dataframe, **kwargs)
        return label_times


def make_windows()



# manualyl pass list of cutoff times



# compose = cp.LabelByWindow(instance_column="customer_id",
#                            time_column="datetime",
#                            labeling_function=make_label,
#                            window_size="30 days",
#                            uses_cutoff_time=False, #if true passes cutoff_time kwargs
#                            drop_null_labels=True)
