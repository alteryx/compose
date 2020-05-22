import json
import os

import pandas as pd

from .object import LabelTimes


def load_label_times(path, df):
    """Read the settings in json format from disk.

    Args:
        path (str) : Directory on disk to read from.
    """
    file = os.path.join(path, 'settings.json')
    assert os.path.exists(file), "settings not found: '%s'" % file

    with open(file, 'r') as file:
        settings = json.load(file)

    lt = LabelTimes(
        data=df.astype(settings['dtypes']),
        target_entity=settings['target_entity'],
        target_types=settings['target_types'],
        name=settings['label_name'],
        search_settings=settings['search_settings'],
        transforms=settings['transforms'],
    )

    return lt


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
    df = pd.read_csv(file)
    lt = load_label_times(path, df)
    return lt


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
    df = pd.read_parquet(file)
    lt = load_label_times(path, df)
    return lt


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
    df = pd.read_pickle(file)
    lt = load_label_times(path, df)
    return lt
