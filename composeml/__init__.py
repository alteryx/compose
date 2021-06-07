# flake8:noqa
from pkg_resources import iter_entry_points

from . import demos
from .label_maker import LabelMaker
from .label_times import LabelTimes, read_label_times
from .version import __version__

for entry_point in iter_entry_points('alteryx_open_src_initialize'):
    try:
        method = entry_point.load()
        if callable(method):
            method('composeml')
    except Exception:
        pass
