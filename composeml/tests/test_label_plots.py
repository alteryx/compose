def test_count_by_time_categorical(total_spent):
    labels = range(2)
    total_spent = total_spent.bin(2, labels=labels)
    ax = total_spent.plot.count_by_time()
    assert ax.get_title() == 'Label Count vs. Cutoff Times'


def test_count_by_time_continuous(total_spent):
    ax = total_spent.plot.count_by_time()
    assert ax.get_title() == 'Label vs. Cutoff Times'


def test_distribution_categorical(total_spent):
    ax = total_spent.bin(2, labels=range(2))
    ax = ax.plot.dist()
    assert ax.get_title() == 'Label Distribution'


def test_distribution_continuous(total_spent):
    ax = total_spent.plot.dist()
    assert ax.get_title() == 'Label Distribution'
