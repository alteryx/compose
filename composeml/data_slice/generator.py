import composeml.data_slice.extension
import pandas as pd

from composeml.data_slice.offset import to_offset
from composeml.utils import can_be_type


class DataSliceGenerator:
    def __init__(self, window_size, gap=None, min_data=None, drop_empty=True):
        self._set_window_size(window_size)
        self.gap = to_offset(gap or self.window_size)
        self.drop_empty = drop_empty
        self.min_data = min_data

    def __call__(self, df):
        data_slices = df.slice(
            size=self.window_size,
            start=self.min_data,
            step=self.gap,
            drop_empty=self.drop_empty,
        )

        for ds in data_slices:
            yield ds

    def _set_window_size(self, window_size):
        """Set and format initial window size parameter.

        Args:
            window_size (str or int): Duration of each data slice.
                The default value for window size is all future data.
        """
        if window_size is not None:
            window_size = to_offset(window_size)

        self.window_size = window_size
