import composeml.data_slice.extension
import pandas as pd

from composeml.data_slice.offset import to_offset


class DataSliceGenerator:
    def __init__(self, window_size, gap=None, min_data=None, drop_empty=True):
        self.window_size = window_size
        self.gap = gap
        self.min_data = min_data
        self.drop_empty = drop_empty

    def _slice_by_time(self, df):
        info = "data frame must be sorted chronologically"
        assert df.index.is_monotonic_increasing, info

        data_slices = df.slice(
            size=self.window_size,
            start=self.min_data,
            step=self.gap,
            drop_empty=self.drop_empty,
        )

        for ds in data_slices:
            yield ds
