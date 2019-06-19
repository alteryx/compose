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
        labels = self.assign(count=1)
        labels = labels.groupby(self.settings['name'])
        distribution = labels['count'].count()
        return distribution

    def _plot_distribution(self, **kwargs):
        plot = self.distribution.plot(kind='bar', **kwargs)
        plot.set_title('label_distribution')
        plot.set_ylabel('count')
        return plot

    @property
    def count_by_time(self):
        labels = self.assign(count=1)

        labels = labels.sort_values('time')
        keys = [self.settings['name'], 'time']
        labels = labels.set_index(keys)

        labels = labels.groupby(keys[0])
        count = labels['count'].cumsum()
        return count

    def _plot_count_by_time(self, **kwargs):
        count = self.count_by_time
        count = count.unstack(self.settings['name'])
        count = count.ffill()

        plot = count.plot(kind='area', **kwargs)
        plot.set_title('count_by_time')
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
        labels['time'] = labels['time'].sub(pd.Timedelta(lead))

        if not inplace:
            return labels

    def bin(self, bins, quantiles=False, labels=None, right=True):
        """
        Bin labels into discrete intervals.

        Args:
            bins (int or array) : The criteria to bin by.

                * bins (int) : Number of bins either equal-width or quantile-based.
                    If `quantiles` is `False`, defines the number of equal-width bins.
                    The range is extended by .1% on each side to include the minimum and maximum values.
                    If `quantiles` is `True`, defines the number of quantiles (e.g. 10 for deciles, 4 for quartiles, etc.)
                * bins (array) : Bin edges either user defined or quantile-based.
                    If `quantiles` is `False`, defines the bin edges allowing for non-uniform width. No extension is done.
                    If `quantiles` is `True`, defines the bin edges usings an array of quantiles (e.g. [0, .25, .5, .75, 1.] for quartiles)

            quantiles (bool) : Determines whether to use a quantile-based discretization function.
            labels (array) : Specifies the labels for the returned bins. Must be the same length as the resulting bins.
            right (bool) : Indicates whether bins includes the rightmost edge or not. Does not apply to quantile-based bins.

        Returns:
            LabelTimes : Instance of labels.

        Examples:
            .. _equal-widths:

            Using bins of `equal-widths`_:

            >>> labels.bin(2).head()
                                            my_labeling_function
            customer_id time                                    
            1           2014-01-01 00:45:00      (157.5, 283.46]
                        2014-01-01 00:48:00      (31.288, 157.5]
            2           2014-01-01 00:01:00      (157.5, 283.46]
                        2014-01-01 00:04:00      (31.288, 157.5]

            .. _custom-widths:

            Using bins of `custom-widths`_:

            >>> bins = [0, 200, 400]
            >>> labels.bin(bins).head()
                                            my_labeling_function
            customer_id time                                    
            1           2014-01-01 00:45:00           (200, 400]
                        2014-01-01 00:48:00             (0, 200]
            2           2014-01-01 00:01:00           (200, 400]
                        2014-01-01 00:04:00             (0, 200]

            .. _quantile-based:

            Using `quantile-based`_ bins:

            >>> labels.bin(4, quantiles=True).head() # (i.e. quartiles)
                                                     my_labeling_function
            customer_id time                                             
            1           2014-01-01 00:45:00             (137.44, 241.062]
                        2014-01-01 00:48:00              (43.848, 137.44]
            2           2014-01-01 00:01:00             (241.062, 283.46]
                        2014-01-01 00:04:00  (31.538999999999998, 43.848]

            .. _labels:

            Assigning `labels`_ to bins:

            >>> labels.bin(3, labels=['low', 'medium', 'high']).head()
                                            my_labeling_function
            customer_id time                                    
            1           2014-01-01 00:45:00                 high
                        2014-01-01 00:48:00                  low
            2           2014-01-01 00:01:00                 high
                        2014-01-01 00:04:00                  low
        """ # noqa
        data = self.copy()
        name = data.settings['name']

        if quantiles:
            data[name] = pd.qcut(data[name].values, q=bins, labels=labels)

        else:
            data[name] = pd.cut(data[name].values, bins=bins, labels=labels, right=right)

        return data
