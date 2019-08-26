import pytest

from .. import demos


@pytest.fixture
def transactions():
    return demos.load_transactions()


def test_transactions(transactions):
    assert len(transactions) == 100
