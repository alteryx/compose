import pandas as pd


class LabelTimes(pd.DataFrame):
    """
    A data frame containing labels made by a label maker.

    Attributes:
        name
        target_entity
        transforms
    """
    _metadata = ['name', 'target_entity', 'settings', 'transforms']

    def __init__(self, data=None, name=None, target_entity=None, settings=None, transforms=None, *args, **kwargs):
        super().__init__(data=data, *args, **kwargs)

        self.name = name
        self.target_entity = target_entity
        self.settings = settings or {}
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
        count = count.sort_values('cutoff_time')
        count = count.set_index([self.name, 'cutoff_time'])
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
        print('Label Distribution\n' + '-' * 18, end='\n')
        distribution = self[self.name].value_counts()
        distribution.index = distribution.index.astype('str')
        distribution['Total:'] = distribution.sum()
        print(distribution.to_string(), end='\n\n\n')

        print('Settings\n' + '-' * 8, end='\n')
        settings = pd.Series(self.settings)

        if settings.empty:
            print('No settings', end='\n\n\n')
        else:
            print(settings.to_string(), end='\n\n\n')

        print('Transforms\n' + '-' * 10, end='\n')
        for step, transform in enumerate(self.transforms):
            transform = pd.Series(transform)
            name = transform.pop('__name__')
            transform = transform.add_prefix('  - ')
            transform = transform.add_suffix(':')
            transform = transform.to_string()
            header = '{}. {}\n'.format(step + 1, name)
            print(header + transform, end='\n\n')

        if len(self.transforms) == 0:
            print('No transforms applied', end='\n\n')

    def copy(self):
        """
        Makes a copy of this instance.

        Returns:
            labels (LabelTimes) : Copy of labels.
        """
        labels = super().copy()
        labels.transforms = labels.transforms.copy()
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

        transform = {'__name__': 'threshold', 'value': value}
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
        labels['cutoff_time'] = labels['cutoff_time'].sub(pd.Timedelta(value))

        transform = {'__name__': 'apply_lead', 'value': value}
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
            cutoff_time           2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function      (157.5, 283.46]      (31.288, 157.5]

            .. _custom-widths:

            Using bins of `custom-widths`_:

            >>> values = labels.bin([0, 200, 400])
            >>> values.head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            cutoff_time           2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function           (200, 400]             (0, 200]

            .. _quantile-based:

            Using `quantile-based`_ bins:

            >>> values = labels.bin(4, quantiles=True) # (i.e. quartiles)
            >>> values.head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            cutoff_time           2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function    (137.44, 241.062]     (43.848, 137.44]

            .. _labels:

            Assigning `labels`_ to bins:

            >>> values = labels.bin(3, labels=['low', 'medium', 'high'])
            >>> values.head(2).T
            label_id                                0                    1
            customer_id                             1                    1
            cutoff_time           2014-01-01 00:45:00  2014-01-01 00:48:00
            my_labeling_function                 high                  low
        """ # noqa
        label_times = self.copy()
        values = label_times[self.name].values

        if quantiles:
            label_times[self.name] = pd.qcut(values, q=bins, labels=labels)

        else:
            label_times[self.name] = pd.cut(values, bins=bins, labels=labels, right=right)

        transform = {
            '__name__': 'bin',
            'bins': bins,
            'quantiles': quantiles,
            'labels': labels,
            'right': right,
        }

        label_times.transforms.append(transform)
        return label_times

    def sample(self, n=None, frac=None, random_state=None):
        """
        Return a random sample of labels.

        Args:
            n (int or dict) : Sample number of labels. A dictionary returns
                the number of samples to each label. Cannot be used with frac.
            frac (float or dict) : Sample fraction of labels. A dictionary returns
                the sample fraction to each label. Cannot be used with n.
            random_state (int) : Seed for the random number generator.

        Returns:
            LabelTimes : Random sample of labels.

        Examples:

            Create mock data:

            >>> labels = {'labels': list('AABBBAA')}
            >>> labels = LabelTimes(labels, name='labels')
            >>> labels
              labels
            0      A
            1      A
            2      B
            3      B
            4      B
            5      A
            6      A

            Sample number of labels:

            >>> labels.sample(n=3, random_state=0)
              labels
            6      A
            2      B
            1      A

            Sample number per label:

            >>> n_per_label = {'A': 1, 'B': 2}
            >>> labels.sample(n=n_per_label, random_state=0)
              labels
            5      A
            4      B
            3      B

            Sample fraction of labels:

            >>> labels.sample(frac=.4, random_state=2)
              labels
            4      B
            1      A
            3      B

            Sample fraction per label:

            >>> frac_per_label = {'A': .5, 'B': .34}
            >>> labels.sample(frac=frac_per_label, random_state=2)
              labels
            5      A
            6      A
            4      B
        """ # noqa
        if isinstance(n, int):
            sample = super().sample(n=n, random_state=random_state)
            return sample

        if isinstance(n, dict):
            sample_per_label = []
            for label, n, in n.items():
                label = self[self[self.name] == label]
                sample = label.sample(n=n, random_state=random_state)
                sample_per_label.append(sample)

            labels = pd.concat(sample_per_label, axis=0, sort=False)
            return labels

        if isinstance(frac, float):
            sample = super().sample(frac=frac, random_state=random_state)
            return sample

        if isinstance(frac, dict):
            sample_per_label = []
            for label, frac, in frac.items():
                label = self[self[self.name] == label]
                sample = label.sample(frac=frac, random_state=random_state)
                sample_per_label.append(sample)

            labels = pd.concat(sample_per_label, axis=0, sort=False)
            return labels
