import featuretools as ft
import pytest

from composeml import LabelMaker


def total_spent(df):
    total = df.amount.sum()
    return total


@pytest.fixture
def labels():
    df = ft.demo.load_mock_customer(return_single_table=True, random_seed=0)
    df = df[["transaction_time", "customer_id", "amount"]]
    df.sort_values("transaction_time", inplace=True)

    lm = LabelMaker(
        target_dataframe_name="customer_id",
        time_index="transaction_time",
        labeling_function=total_spent,
        window_size="1h",
    )

    lt = lm.search(
        df,
        minimum_data="10min",
        num_examples_per_instance=2,
        gap="30min",
        drop_empty=True,
        verbose=False,
    )

    lt = lt.threshold(1250)
    return lt


def test_dfs(labels):
    target_column = labels.target_columns[0]
    es = ft.demo.load_mock_customer(return_entityset=True, random_seed=0)
    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        cutoff_time=labels,
        cutoff_time_in_index=True,
    )
    assert target_column in feature_matrix

    columns = ["customer_id", "time", target_column]
    given_labels = feature_matrix.reset_index()[columns]
    given_labels = given_labels.sort_values(["customer_id", "time"])
    given_labels = given_labels.reset_index(drop=True)
    given_labels = given_labels.rename_axis("label_id")

    assert given_labels.equals(labels)
