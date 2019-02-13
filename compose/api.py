from compose import LabelByInstance

# label all events
# returns <instance id, label, time>
def make_label_times(all_events, threshold=0):
    return label_times

compose = LabelByInstance(instance_column="customer_id",
                          labeling_function=make_label_times)

label_times = compose.label(df, threshold=10)



import composeml as cp

# label windows return label
# label time and instance id implicit
def make_label(window_events, threshold):
    label = events["amount"].sum() > threshold
    return label


compose = cp.LabelByWindow(instance_column="customer_id",
                           time_column="datetime",
                           labeling_function=make_label,
                           window_size="30 days",
                           uses_cutoff_time=False, #if true passes cutoff_time kwargs
                           drop_null_labels=True)

search_params = {
    "min_data": ,
    "num_examples_per_instance": ,
    "window_size": "",
    "start_from": "",
    "cutoff_times": "",
}

label_times = compose.label(df, threshold=10)


# example with training_window
# label time and instance id implicit
def make_label(window_events, cutoff_time, threshold):
    label = events["amount"].sum() > threshold
    return label


compose = cp.LabelByWindow(instance_column="customer_id",
                           time_column="datetime",
                           labeling_function=make_label,
                           window_size="30 days",
                           training_data=

search_params = {
    "min_data": ,
    "num_examples_per_instance": ,
    "window_size": "",
    "start_from": ""
}

label_times = compose.label(df, threshold=10)




"""
Design notes

* Need time delta object
* label time transforms - sample, filter,
* subclass dataframe to be labeltimes
* make a nice summarize function
# should window size  go constructor or search params


Misc
- what to name library
- what to important as

"""


