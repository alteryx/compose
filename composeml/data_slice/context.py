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
