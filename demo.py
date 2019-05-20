from composeml import LabelMaker
from featuretools.demo import load_mock_customer

full_df = load_mock_customer(return_single_table=True)


def my_labeling_function(df_slice):
    label = df_slice["amount"].mean() > 80
    return label


lm = LabelMaker(
    target_entity="customer_id",
    time_index="transaction_time",
    labeling_function=my_labeling_function,
    window_size="2h",
)

lt = lm.search(
    full_df,
    minimum_data="1h",
    num_examples_per_instance=2,
    gap="2h",
)

lt