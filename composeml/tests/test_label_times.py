from pytest import raises

from composeml.label_times import LabelTimes
from composeml.tests.utils import to_csv


def test_count_by_time_categorical(total_spent):
    given_answer = total_spent.bin(2, labels=range(2))
    given_answer = to_csv(given_answer.count_by_time)

    answer = [
        "time,0,1",
        "2019-01-01 08:00:00,0,1",
        "2019-01-01 08:30:00,0,2",
        "2019-01-01 09:00:00,0,3",
        "2019-01-01 09:30:00,0,4",
        "2019-01-01 10:00:00,0,5",
        "2019-01-01 10:30:00,1,5",
        "2019-01-01 11:00:00,2,5",
        "2019-01-01 11:30:00,3,5",
        "2019-01-01 12:00:00,4,5",
        "2019-01-01 12:30:00,5,5",
    ]

    assert given_answer == answer


def test_count_by_time_continuous(total_spent):
    given_answer = total_spent.count_by_time
    given_answer = to_csv(given_answer, header=True, index=True)

    answer = [
        "time,total_spent",
        "2019-01-01 08:00:00,1",
        "2019-01-01 08:30:00,2",
        "2019-01-01 09:00:00,3",
        "2019-01-01 09:30:00,4",
        "2019-01-01 10:00:00,5",
        "2019-01-01 10:30:00,6",
        "2019-01-01 11:00:00,7",
        "2019-01-01 11:30:00,8",
        "2019-01-01 12:00:00,9",
        "2019-01-01 12:30:00,10",
    ]

    assert given_answer == answer


def test_sorted_distribution(capsys, total_spent):
    bins = [0, 5, 10, 20]
    total_spent.bin(bins).describe()
    captured = capsys.readouterr()

    out = "\n".join(
        [
            "Label Distribution",
            "------------------",
            "(0, 5]      5",
            "(5, 10]     4",
            "(10, 20]    0",
            "Total:      9",
            "",
            "",
            "Settings",
            "--------",
            "num_examples_per_instance             -1",
            "target_column                total_spent",
            "target_dataframe_name        customer_id",
            "target_type                     discrete",
            "",
            "",
            "Transforms",
            "----------",
            "1. bin",
            "  - bins:         [0, 5, 10, 20]",
            "  - labels:                 None",
            "  - precision:                 3",
            "  - quantiles:             False",
            "  - right:                  True",
            "",
            "",
        ]
    )

    assert captured.out == out


def test_describe_no_transforms(capsys):
    data = {"target": range(3)}
    LabelTimes(data).describe()
    captured = capsys.readouterr()
    out = "\n".join(
        [
            "Label Distribution",
            "------------------",
            "count    3.0",
            "mean     1.0",
            "std      1.0",
            "min      0.0",
            "25%      0.5",
            "50%      1.0",
            "75%      1.5",
            "max      2.0",
            "",
            "",
            "Settings",
            "--------",
            "target_column                target",
            "target_dataframe_name          None",
            "target_type              continuous",
            "",
            "",
            "Transforms",
            "----------",
            "No transforms applied",
            "",
            "",
        ]
    )

    assert captured.out == out


def test_distribution_categorical(total_spent):
    labels = range(2)
    given_answer = total_spent.bin(2, labels=labels).distribution
    given_answer = to_csv(given_answer)

    answer = [
        "total_spent,count",
        "0,5",
        "1,5",
    ]

    assert given_answer == answer


def test_distribution_continous(total_spent):
    distribution = total_spent.distribution
    actual = to_csv(distribution.round(4))

    expected = [
        ",total_spent",
        "count,10.0",
        "mean,4.5",
        "std,3.0277",
        "min,0.0",
        "25%,2.25",
        "50%,4.5",
        "75%,6.75",
        "max,9.0",
    ]

    assert actual == expected


def test_target_type(total_spent):
    types = total_spent.target_types
    assert types["total_spent"] == "continuous"
    total_spent = total_spent.threshold(5)
    types = total_spent.target_types
    assert types["total_spent"] == "discrete"


def test_count(total_spent):
    given_answer = total_spent.count
    given_answer = to_csv(given_answer, index=True)

    answer = [
        "customer_id,count",
        "0,2",
        "1,3",
        "2,4",
        "3,1",
    ]

    assert given_answer == answer


def test_label_select_errors(total_spent):
    match = "only one target exists"
    with raises(AssertionError, match=match):
        total_spent.select("a")

    lt = total_spent.copy()
    lt.target_columns.append("b")

    match = "target name must be string"
    with raises(TypeError, match=match):
        total_spent.select(123)

    match = 'target "a" not found'
    with raises(AssertionError, match=match):
        lt.select("a")
