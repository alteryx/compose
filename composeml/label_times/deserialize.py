import json
import os

import pandas as pd

from .object import LabelTimes


def read_config(path):
    """Reads config file from disk."""
    file = os.path.join(path, 'settings.json')
    assert os.path.exists(file), "settings not found: '%s'" % file

    with open(file, 'r') as file:
        settings = json.load(file)
        return settings


def read_data(path):
    """Reads data file from disk."""
    file = ''
    for file in os.listdir(path):
        if file.startswith('data'): break

    assert file.startswith('data'), "data not found"
    extension = os.path.splitext(file)[1].lstrip('.')
    info = 'file extension must be csv, parquet, or pickle'
    assert extension in ['csv', 'parquet', 'pickle'], info

    read = getattr(pd, 'read_%s' % extension)
    data = read(os.path.join(path, file))
    return data


def read_label_times(path, load_settings=True):
    """Reads label times from disk.

    Args:
        path (str): Directory where label times is stored.

    Returns:
        lt (LabelTimes): Deserialized label times.
    """
    kwargs = {}
    data = read_data(path)

    if load_settings:
        config = read_config(path)
        data = data.astype(config['dtypes'])
        kwargs.update(config['label_times'])

    lt = LabelTimes(data=data, **kwargs)
    return lt
