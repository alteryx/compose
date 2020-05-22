import pandas as pd


class DataSliceContext:
    """Metadata for data slice."""

    def __init__(self, gap=None, window=None, slice_number=None, target_entity=None, target_instance=None):
        """Metadata for data slice.

        Args:
            gap (tuple) : Start and stop time for gap.
            window (tuple) : Start and stop time for window.
            slice_number (int) : Slice number.
            target_entity (int) : Target entity.
            target_instance (int) : Target instance.
        """
        self.gap = gap or (None, None)
        self.window = window or (None, None)
        self.slice_number = slice_number
        self.target_entity = target_entity
        self.target_instance = target_instance


class DataSlice(pd.DataFrame):
    """Data slice for labeling function."""
    _metadata = ['context']

    @property
    def _constructor(self):
        return DataSlice

    def __str__(self):
        """Metadata of data slice."""
        info = {
            'slice_number': self.context.slice_number,
            self.context.target_entity: self.context.target_instance,
            'window': '[{}, {})'.format(*self.context.window),
            'gap': '[{}, {})'.format(*self.context.gap),
        }

        info = pd.Series(info).to_string()
        return info
