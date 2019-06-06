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

    @property
    def distribution(self):
        name = self.settings['name']
        target_entity = self.settings['target_entity']

        labels = self.assign(count=1).groupby([target_entity, name])
        distribution = labels['count'].count()
        return distribution

    def _plot_distribution(self, **kwargs):
        name = self.settings['name']
        distribution = self.distribution.unstack(name)

        plot = distribution.plot(kind='bar', **kwargs)
        plot.set_title('label_distribution')
        plot.set_ylabel('count')
        return plot

    @property
    def count_by_time(self):
        target_entity = self.settings['target_entity']
        labels = self.assign(count=1).sort_index(level='time')
        labels = labels.groupby(target_entity)
        count = labels['count'].cumsum()
        return count

    def _plot_count_by_time(self, **kwargs):
        target_entity = self.settings['target_entity']
        count_by_time = self.count_by_time.unstack(target_entity).ffill()

        plot = count_by_time.plot(kind='area', **kwargs)
        plot.set_title(self.settings['name'])
        plot.set_ylabel('count')
        return plot

    def _with_plots(self):
        self.plot.count_by_time = self._plot_count_by_time
        self.plot.distribution = self._plot_distribution
        return self

    def describe(self):
        """Prints out label distribution and the settings used to make the labels."""
        labels = self[self.settings['name']]
        label_counts = labels.value_counts()
        total = 'Total number of labels:  {}'
        total = total.format(label_counts.sum())

        print(total, end='\n\n')
        print(label_counts, end='\n\n')
        print(self.distribution, end='\n\n')
        print(pd.Series(self.settings), end='\n\n')

    def copy(self):
        """
        Makes a copy of this instance.

        Returns:
            labels (LabelTimes) : Copy of labels.
        """
        labels = super().copy()
        labels.settings = self.settings.copy()
        return labels._with_plots()

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

        if not inplace:
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

        if not inplace:
            return labels

    def bin(self, bins, labels=None, right=True, quantiles=False):
        """
        Bin labels into discrete intervals.

        Args:
            bins (int or array) : The criteria to bin by.

                * int : Number of bins either equal-width or quantile-based.
                    If `quantiles` is `False`, defines the number of equal-width bins.
                    The range is extended by .1% on each side to include the minimum and maximum values.
                    If `quantiles` is `True`, defines the number of quantiles (e.g. 10 for deciles, 4 for quartiles, etc.)
                * array : Bin edges as defined values or quantiles.
                    If `quantiles` is `False`, defines the bin edges allowing for non-uniform width. No extension is done.
                    If `quantiles` is `True`, defines the bin edges usings an array of quantiles (e.g. [0, .25, .5, .75, 1.] for quartiles)

            labels (array) : Specifies the labels for the returned bins. Must be the same length as the resulting bins.
            right (bool) : Indicates whether bins includes the rightmost edge or not. Does not apply to quantile-based bins.
            quantiles (bool) : Determines whether to use a quantile-based discretization function.

        Returns:
            LabelTimes : Instance of labels.
        """
        data = self.copy()
        name = data.settings['name']

        if quantiles:
            data[name] = pd.qcut(data[name].values, q=bins, labels=labels)

        else:
            data[name] = pd.cut(data[name].values, bins=bins, labels=labels, right=right)

        return data
