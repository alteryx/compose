from io import StringIO

import pandas as pd


def read_csv(data, **kwargs):
    if isinstance(data, list):
        data = '\n'.join(data)

    with StringIO(data) as data:
        df = pd.read_csv(data, **kwargs)

    return df


def to_csv(label_times):
    df = pd.DataFrame(label_times)
    csv = df.to_csv(index=False)
    return csv.splitlines()
