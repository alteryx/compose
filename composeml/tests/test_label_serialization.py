import os
import shutil

import pytest

from composeml.label_times import read_csv


@pytest.fixture
def path():
    pwd = os.path.dirname(__file__)
    path = os.path.join(pwd, '.cache')
    yield path
    shutil.rmtree(path)


def test_to_csv(path, total_spent):
    total_spent.to_csv(path)

    for filename in ['label_times.csv', 'settings.json', 'transforms.json']:
        file = os.path.join(path, filename)
        assert os.path.exists(file)


# def test_read_csv(path, total_spent):
#     total_spent.to_csv(path)
#     total_spent_copy = read_csv(path)
#     assert total_spent.equals(total_spent_copy)
