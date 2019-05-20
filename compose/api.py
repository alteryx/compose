import composeml as cp


def my_labeling_function(df_slice):
    """one slice of data inside of the prediction window for single instance of target entity"""
    label = df_slice["voltage"].mean()
    return label

# todo name
lm = cp.LabelMaker(target_entity="machine_id",
                   time_index="timestamp",
                   labeling_function=my_labeling_function,
                   window_size="7 days")

# describe the parameters to search for the labels
# returns a LabelTimes object, which is basically a pandas dataframe
lt = lm.search(dataframe=full_df,
               minimum_data="20 days", # minimum data before starting search
               num_examples_per_instance=10, # examples per unique instance of target entity
               gap="7 days") # time between examples

lt.summarize() # prints out distribution of labels

lt.describe() # prints out all the settings used to make the labels

# functions to modify label times and return a new copy of label times
lt2 = lt.threshold(10) # applies to label column
lt3 = lt.apply_lead("7 days") # applies to time column


# utilities to debug labeling functions

# return a sample slice of data to help user debug
# can take any of the arguments of search
df_slice = lm.sample_slice(full_df, random_seed=5)
my_labeling_function(df_slice)

"""Implementation Notes

* subclass dataframe to be labeltimes (https://pandas.pydata.org/pandas-docs/stable/development/extending.html#subclassing-pandas-data-structures)

"""




