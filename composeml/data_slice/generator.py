from composeml.data_slice.extension import DataSliceContext, DataSliceFrame


class DataSliceGenerator:
    """Generates data slices for the lable maker."""
    def __init__(self, window_size, gap=None, min_data=None, drop_empty=True):
        self.window_size = window_size
        self.gap = gap
        self.min_data = min_data
        self.drop_empty = drop_empty

    def __call__(self, df):
        """Applies the data slice generator to the data frame."""
        is_column = self.window_size in df
        method = 'column' if is_column else 'time'
        attr = '_slice_by_%s' % method
        return getattr(self, attr)(df)

    def _slice_by_column(self, df):
        slices = df.groupby(self.window_size, sort=False)
        slice_number = 1

        for group, ds in slices:
            ds = DataSliceFrame(ds)
            ds.context = DataSliceContext(
                slice_number=slice_number,
                slice_start=ds.index[0],
                slice_stop=ds.index[-1],
            )
            slice_number += 1
            del ds.context.next_start
            setattr(ds.context, self.window_size, group)
            yield ds

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
