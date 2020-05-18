import os

import pandas as pd

from .data_frame import LabelTimes


def read_csv(path, filename='label_times.csv', load_settings=True):
    """Read label times in csv format from disk.

    Args:
        path (str) : Directory on disk to read from.
        filename (str) : Filename for label times. Default value is `label_times.csv`.
        load_settings (bool) : Whether to load the settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = os.path.join(path, filename)
    assert os.path.exists(file), "data not found: '%s'" % file

    data = pd.read_csv(file, index_col='id')
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times


def read_parquet(path, filename='label_times.parquet', load_settings=True):
    """Read label times in parquet format from disk.

    Args:
        path (str) : Directory on disk to read from.
        filename (str) : Filename for label times. Default value is `label_times.parquet`.
        load_settings (bool) : Whether to load the settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = os.path.join(path, filename)
    assert os.path.exists(file), "data not found: '%s'" % file

    data = pd.read_parquet(file)
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times


def read_pickle(path, filename='label_times.pickle', load_settings=True):
    """Read label times in parquet format from disk.

    Args:
        path (str) : Directory on disk to read from.
        filename (str) : Filename for label times. Default value is `label_times.parquet`.
        load_settings (bool) : Whether to load the settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = os.path.join(path, filename)
    assert os.path.exists(file), "data not found: '%s'" % file

    data = pd.read_pickle(file)
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times
