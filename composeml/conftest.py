# flake8:noqa
import pytest

from composeml.tests.test_label_times import labels


@pytest.fixture(autouse=True)
def add_labels(doctest_namespace, labels):
    doctest_namespace['labels'] = labels
