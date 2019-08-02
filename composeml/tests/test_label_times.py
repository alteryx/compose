def test_describe(labels):
    labels = labels.bin(2)
    labels.settings.update(num_examples_per_instance=2)
    assert labels.describe() is None


def test_describe_empty(labels):
    labels.settings.clear()
    assert labels.describe() is None
