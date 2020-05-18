import pandas as pd


class DataFrame(pd.DataFrame):
    """A data frame containing labels made by a label maker.

    Attributes:
        settings
    """
    _metadata = [
        '_cache',
        'label_name',
        'label_type',
        'target_entity',
        'transforms',
    ]

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other label times data frames.

        Args:
            other (LabelTimes) : The label times from which to get the attributes from.
            method (str) : A passed method name for optionally taking different types of propagation actions based on this value.
        """
        if method == 'concat':
            other = other.objs[0]

            for key in self._metadata:
                value = getattr(other, key, None)
                setattr(self, key, value)

            return self

        return super().__finalize__(other=other, method=method, **kwargs)

    @property
    def _constructor(self):
        return type(self)
