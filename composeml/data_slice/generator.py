import composeml.data_slice.extension  # noqa


class DataSliceGenerator:
    """Generates data slices for the label maker."""
    def __init__(self, window_size, gap=None, min_data=None, drop_empty=True):
        self.window_size = window_size
        self.gap = gap
        self.min_data = min_data
        self.drop_empty = drop_empty

    def __call__(self, df):
        """Applies the data slice generator to the data frame."""
        return self._slice_by_time(df)

    def _slice_by_time(self, df):
        """Slices data along the time index."""
        data_slices = df.slice(
            size=self.window_size,
            start=self.min_data,
            step=self.gap,
            drop_empty=self.drop_empty,
        )

        for ds in data_slices:
            yield ds
