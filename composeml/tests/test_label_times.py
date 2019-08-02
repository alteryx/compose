from ..label_times import LabelTimes


def test_describe(labels):
    labels = labels.bin(2)
    labels.settings.update(num_examples_per_instance=2)
    assert labels.describe() is None


def test_describe_empty():
    data = {'labels': [1, 2]}
    label_times = LabelTimes(data, name="labels")
    label_times.describe()
