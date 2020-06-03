import os

import pandas as pd

from .object import LabelTimes


def read_label_times(path, load_settings=True):
    """Read label times in csv format from disk.

    Args:
        path (str): Directory where label times is stored.
        load_settings (bool): Whether to load settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = ''
    for file in os.listdir(path):
        if file.startswith('data'): break

    assert file.startswith('data'), "data not found"
    extension = os.path.splitext(file)[1].lstrip('.')
    info = 'file extension must be csv, parquet, or pickle'
    assert extension in ['csv', 'parquet', 'pickle'], info

    read = getattr(pd, 'read_%s' % extension)
    data = read(os.path.join(path, file))
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times
