import composeml as cp
from featuretools.demo import load_mock_customer


def my_labeling_function(df_slice):
    label = df_slice["amount"].mean()
    return label


lm = cp.LabelMaker(
    target_entity="customer_id",
    time_index="transaction_time",
    labeling_function=my_labeling_function,
    window_size="7 days",
)

full_df = load_mock_customer(return_single_table=True)

lt = lm.search(
    dataframe=full_df,
    minimum_data="20 days",
    num_examples_per_instance=10,
    gap="7 days",
)
