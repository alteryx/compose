import pytest

from .. import datasets


@pytest.fixture
def transactions():
    return datasets.transactions()


def test_transactions(transactions):
    assert len(transactions) == 150
