from io import StringIO

import pandas as pd


def read_csv(data, **kwargs):
    if isinstance(data, list):
        data = '\n'.join(data)

    with StringIO(data) as data:
        df = pd.read_csv(data, **kwargs)

    return df


def to_csv(label_times, **kwargs):
    df = pd.DataFrame(label_times)
    csv = df.to_csv(**kwargs)
    return csv.splitlines()
