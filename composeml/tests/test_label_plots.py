from pytest import raises


def test_count_by_time_categorical(total_spent):
    total_spent = total_spent.bin(2, labels=range(2))
    title = total_spent.plot.count_by_time().get_title()
    assert title == "Label Count vs. Cutoff Times"


def test_count_by_time_continuous(total_spent):
    title = total_spent.plot.count_by_time().get_title()
    assert title == "Label vs. Cutoff Times"


def test_distribution_categorical(total_spent):
    ax = total_spent.bin(2, labels=range(2))
    title = ax.plot.dist().get_title()
    assert title == "Label Distribution"


def test_distribution_continuous(total_spent):
    title = total_spent.plot.dist().get_title()
    assert title == "Label Distribution"


def test_single_target(total_spent):
    lt = total_spent.copy()
    lt.target_columns.append("target_2")
    match = "must first select an individual target"

    with raises(AssertionError, match=match):
        lt.plot.dist()

    with raises(AssertionError, match=match):
        lt.plot.count_by_time()
