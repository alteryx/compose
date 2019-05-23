import pandas as pd


class LabelTimes(pd.DataFrame):
    """
    A data frame containing labels made by a label maker.

    Attributes:
        settings
    """
    _metadata = ['_settings']

    @property
    def _constructor(self):
        return LabelTimes

    @property
    def settings(self):
        if not hasattr(self, '_settings'):
            self._settings = {}

        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    def describe(self):
        """Prints out label distribution and the settings used to make the labels."""
        labels = self[self.settings['name']]
        distribution = labels.value_counts()
        print(distribution, end='\n\n')
        print(pd.Series(self.settings), end='\n\n')

    def copy(self):
        """
        Makes a copy of this instance.

        Returns:
            labels (LabelTimes) : Copy of labels.
        """
        labels = super().copy()
        labels.settings = self.settings.copy()
        return labels

    def threshold(self, value, inplace=False):
        """
        Creates binary labels by testing if labels are above threshold.

        Args:
            value (float) : Value of threshold.
            inplace (bool) : Modify labels in place.

        Returns:
            labels (LabelTimes) : Instance of labels.
        """
        labels = self if inplace else self.copy()
        name = labels.settings['name']
        labels[name] = labels[name].gt(value)
        labels.settings.update(threshold=value)
        return labels

    def apply_lead(self, lead, inplace=False):
        """
        Shifts the label times earlier for predicting in advance.

        Args:
            lead (str) : Time to shift earlier.
            inplace (bool) : Modify labels in place.

        Returns:
            labels (LabelTimes) : Instance of labels.
        """
        labels = self if inplace else self.copy()
        labels.settings.update(lead=lead)

        names = labels.index.names
        labels.reset_index(inplace=True)
        labels['time'] = labels['time'].sub(pd.Timedelta(lead))
        labels.set_index(names, inplace=True)

        return labels
