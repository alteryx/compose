def test_describe(labels):
    labels = labels.bin(2)
    assert labels.describe() is None
