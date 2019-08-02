from ..label_times import LabelTimes


def test_describe(labels):
    labels = labels.bin(2)
    assert labels.describe() is None


def test_describe_empty():
    data = {"label": [100, 200]}
    label_times = LabelTimes(data, name="label")
    label_times.describe()
