import pandas as pd


class LabelTimes(pd.DataFrame):
    """
    A data frame containing labels made by a label maker.

    Attributes:
        name
        target_entity
        transforms
    """
    _metadata = ['name', 'target_entity', 'transforms']

    def __init__(self, data, name=None, target_entity=None, transforms=None):
        super().__init__(data=data)
        self.name = name
        self.target_entity = target_entity
        self.transforms = transforms or []

    @property
    def _constructor(self):
        return LabelTimes

    @property
    def distribution(self):
        labels = self.assign(count=1)
        labels = labels.groupby(self.name)
        distribution = labels['count'].count()
        return distribution

    def _plot_distribution(self, **kwargs):
        plot = self.distribution.plot(kind='bar', **kwargs)
        plot.set_title('Label Distribution')
        plot.set_ylabel('count')
        return plot

    @property
    def count_by_time(self):
        count = self.assign(count=1)
        count = count.sort_values('time')
        count = count.set_index([self.name, 'time'])
        count = count.groupby(self.name)
        count = count['count'].cumsum()
        return count

    def _plot_count_by_time(self, **kwargs):
        count = self.count_by_time
        count = count.unstack(self.name)
        count = count.ffill()

        plot = count.plot(kind='area', **kwargs)
        plot.set_title('Label Count vs. Time')
        plot.set_ylabel('count')
        return plot

    def _with_plots(self):
        self.plot.count_by_time = self._plot_count_by_time
        self.plot.distribution = self._plot_distribution
        return self

    def describe(self):
        """Prints out label info with transform settings that reproduce labels."""
        count = self[self.name].value_counts()
        count['total'] = count.sum()
        print(count.to_string(), end='\n\n')

        for transform in self.transforms:
            transform = pd.Series(transform)
            name = transform.pop('name')
            transform = transform.rename_axis(name)
            print(transform.to_string(), end='\n\n')

    def copy(self):
        """
        Makes a copy of this instance.

        Returns:
            labels (LabelTimes) : Copy of labels.
        """
        labels = super().copy()
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
        labels[self.name] = labels[self.name].gt(value)

        transform = {'name': 'threshold', 'value': value}
        labels.transforms.append(transform)

        if not inplace:
            return labels

    def apply_lead(self, value, inplace=False):
        """
        Shifts the label times earlier for predicting in advance.

        Args:
            value (str) : Time to shift earlier.
            inplace (bool) : Modify labels in place.

        Returns:
            labels (LabelTimes) : Instance of labels.
        """
        labels = self if inplace else self.copy()
        labels['time'] = labels['time'].sub(pd.Timedelta(value))

        transform = {'name': 'lead', 'value': value}
        labels.transforms.append(transform)

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

            >>> labels.bin(2).head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            time                  2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function      (157.5, 283.46]      (31.288, 157.5]

            .. _custom-widths:

            Using bins of `custom-widths`_:

            >>> values = labels.bin([0, 200, 400])
            >>> values.head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            time                  2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function           (200, 400]             (0, 200]

            .. _quantile-based:

            Using `quantile-based`_ bins:

            >>> values = labels.bin(4, quantiles=True) # (i.e. quartiles)
            >>> values.head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            time                  2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function    (137.44, 241.062]     (43.848, 137.44]

            .. _labels:

            Assigning `labels`_ to bins:

            >>> values = labels.bin(3, labels=['low', 'medium', 'high'])
            >>> values.head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            time                  2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function                 high                  low
        """ # noqa
        label_times = self.copy()
        values = label_times[self.name].values

        if quantiles:
            label_times[self.name] = pd.qcut(values, q=bins, labels=labels)

        else:
            label_times[self.name] = pd.cut(values, bins=bins, labels=labels, right=right)

        transform = {
            'name': 'bin',
            'bins': bins,
            'quantiles': quantiles,
            'labels': labels,
            'right': right,
        }
        label_times.transforms.append(transform)
        return label_times
