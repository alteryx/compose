import pandas as pd


class DataSlice(pd.DataFrame):
    """Data slice for labeling function."""
    _metadata = ['context']

    @property
    def _constructor(self):
        return DataSlice

    @property
    def _info(self):
        return {
            'slice_number': self.context.slice_number,
            self.context.target_entity: self.context.target_instance,
            'window': '[{}, {})'.format(*self.context.window),
            'gap': '[{}, {})'.format(*self.context.gap),
        }

    def __str__(self):
        """Metadata of data slice."""
        info = pd.Series(self._info)
        return info.to_string()
