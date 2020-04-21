import os

import pandas as pd

PWD = os.path.dirname(__file__)


def test_manifest():
    filepath = os.path.join(PWD, 'test.csv')
    df = pd.read_csv(filepath)
