def test_distribution_plot(labels):
    labels = labels.threshold(200)
    plot = labels.plot.distribution()
    assert plot.get_title() == 'Label Distribution'


def test_count_by_time_plot(labels):
    labels = labels.threshold(200)
    plot = labels.plot.count_by_time()
    assert plot.get_title() == 'Label Count vs. Time'
