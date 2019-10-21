from io import StringIO

import pandas as pd


def read_csv(data, **kwargs):
    """Helper function for creating a dataframe from in-memory CSV string (or list of strings).

    Args:
        data (str or list) : CSV string(s)

    Returns:
        DataFrame : Instance of a dataframe.
    """
    if isinstance(data, list):
        data = '\n'.join(data)

    # This creates a file-like object for reading in CSV string.
    with StringIO(data) as data:
        df = pd.read_csv(data, **kwargs)

    return df


def to_csv(label_times, **kwargs):
    df = pd.DataFrame(label_times)
    csv = df.to_csv(**kwargs)
    return csv.splitlines()
