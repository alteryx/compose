from io import StringIO

import pandas as pd


def read_csv(csv):
    if isinstance(csv, list):
        csv = '\n'.join(csv)

    with StringIO(csv) as file:
        df = pd.read_csv(file)

    return df
