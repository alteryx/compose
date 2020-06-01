import json
import os

import pandas as pd

from composeml.label_plots import LabelPlots


def read_csv(path, filename='label_times.csv', load_settings=True):
    """Read label times in csv format from disk.

    Args:
        path (str) : Directory on disk to read from.
        filename (str) : Filename for label times. Default value is `label_times.csv`.
        load_settings (bool) : Whether to load the settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = os.path.join(path, filename)
    assert os.path.exists(file), "data not found: '%s'" % file

    data = pd.read_csv(file, index_col='id')
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times


def read_parquet(path, filename='label_times.parquet', load_settings=True):
    """Read label times in parquet format from disk.

    Args:
        path (str) : Directory on disk to read from.
        filename (str) : Filename for label times. Default value is `label_times.parquet`.
        load_settings (bool) : Whether to load the settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = os.path.join(path, filename)
    assert os.path.exists(file), "data not found: '%s'" % file

    data = pd.read_parquet(file)
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times


def read_pickle(path, filename='label_times.pickle', load_settings=True):
    """Read label times in parquet format from disk.

    Args:
        path (str) : Directory on disk to read from.
        filename (str) : Filename for label times. Default value is `label_times.parquet`.
        load_settings (bool) : Whether to load the settings used to make the label times.

    Returns:
        LabelTimes : Deserialized label times.
    """
    file = os.path.join(path, filename)
    assert os.path.exists(file), "data not found: '%s'" % file

    data = pd.read_pickle(file)
    label_times = LabelTimes(data=data)

    if load_settings:
        label_times = label_times._load_settings(path)

    return label_times


class LabelTimes(pd.DataFrame):
    """A data frame containing labels made by a label maker.

    Attributes:
        settings
    """
    _metadata = ['settings']

    def __init__(self, data=None, target_entity=None, name=None, label_type=None, settings=None, *args, **kwargs):
        super().__init__(data=data, *args, **kwargs)

        if label_type is not None:
            error = 'label type must be "continuous" or "discrete"'
            assert label_type in ['continuous', 'discrete'], error

        self.settings = settings or {
            'target_entity': target_entity,
            'labeling_function': name,
            'label_type': label_type,
            'transforms': [],
        }

        self.plot = LabelPlots(self)

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other label times.

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
        return LabelTimes

    @property
    def label_name(self):
        """Get name of label times."""
        return self.settings.get('labeling_function')

    @label_name.setter
    def label_name(self, value):
        """Set name of label times."""
        self.settings['labeling_function'] = value

    @property
    def target_entity(self):
        """Get target entity of label times."""
        return self.settings.get('target_entity')

    @target_entity.setter
    def target_entity(self, value):
        """Set target entity of label times."""
        self.settings['target_entity'] = value

    @property
    def label_type(self):
        """Get label type."""
        return self.settings.get('label_type')

    @label_type.setter
    def label_type(self, value):
        """Set label type."""
        self.settings['label_type'] = value

    @property
    def transforms(self):
        """Get transforms of label times."""
        return self.settings.get('transforms', [])

    @transforms.setter
    def transforms(self, value):
        """Set transforms of label times."""
        self.settings['transforms'] = value

    @property
    def is_discrete(self):
        """Whether labels are discrete."""
        if self.label_type is None:
            self.label_type = self.infer_type()

        return self.label_type == 'discrete'

    @property
    def distribution(self):
        """Returns label distribution if labels are discrete."""
        if self.is_discrete:
            labels = self.assign(count=1)
            labels = labels.groupby(self.label_name)
            distribution = labels['count'].count()
            return distribution

    @property
    def count(self):
        """Returns label count per instance."""
        count = self.groupby(self.target_entity)
        count = count[self.label_name].count()
        count = count.to_frame('count')
        return count

    @property
    def count_by_time(self):
        """Returns label count across cutoff times."""
        if self.is_discrete:
            keys = ['time', self.label_name]
            value = self.groupby(keys).time.count()
            value = value.unstack(self.label_name).fillna(0)
        else:
            value = self.groupby('time')
            value = value[self.label_name].count()

        value = value.cumsum()  # In Python 3.5, these values automatically convert to float.
        value = value.astype('int')
        return value

    def describe(self):
        """Prints out label info with transform settings that reproduce labels."""
        if self.label_name is not None and self.is_discrete:
            print('Label Distribution\n' + '-' * 18, end='\n')
            distribution = self[self.label_name].value_counts()
            distribution.index = distribution.index.astype('str')
            distribution.sort_index(inplace=True)
            distribution['Total:'] = distribution.sum()
            print(distribution.to_string(), end='\n\n\n')

        settings = pd.Series(self.settings)
        transforms = settings.pop('transforms')

        print('Settings\n' + '-' * 8, end='\n')

        if settings.isnull().all():
            print('No settings', end='\n\n\n')
        else:
            settings.sort_index(inplace=True)
            print(settings.to_string(), end='\n\n\n')

        print('Transforms\n' + '-' * 10, end='\n')

        for step, transform in enumerate(transforms):
            transform = pd.Series(transform)
            transform.sort_index(inplace=True)
            name = transform.pop('transform')
            transform = transform.add_prefix('  - ')
            transform = transform.add_suffix(':')
            transform = transform.to_string()
            header = '{}. {}\n'.format(step + 1, name)
            print(header + transform, end='\n\n')

        if len(transforms) == 0:
            print('No transforms applied', end='\n\n')

    def copy(self, **kwargs):
        """
        Makes a copy of this object.

        Args:
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.copy method

        Returns:
            LabelTimes : Copy of label times.
        """
        label_times = super().copy(**kwargs)
        label_times.settings = self.settings.copy()
        label_times.transforms = self.transforms.copy()
        return label_times

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
        labels[self.label_name] = labels[self.label_name].gt(value)

        labels.label_type = 'discrete'
        labels.settings['label_type'] = 'discrete'

        transform = {'transform': 'threshold', 'value': value}
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

        transform = {'transform': 'apply_lead', 'value': value}
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
        """  # noqa
        label_times = self.copy()
        values = label_times[self.label_name].values

        if quantiles:
            label_times[self.label_name] = pd.qcut(values, q=bins, labels=labels)

        else:
            label_times[self.label_name] = pd.cut(values, bins=bins, labels=labels, right=right)

        transform = {
            'transform': 'bin',
            'bins': bins,
            'quantiles': quantiles,
            'labels': labels,
            'right': right,
        }

        label_times.transforms.append(transform)
        label_times.label_type = 'discrete'
        return label_times

    def _sample(self, key, value, settings, random_state=None, replace=False):
        """Returns a random sample of labels.

        Args:
            key (str) : Determines the sampling method. Can either be 'n' or 'frac'.
            value (int or float) : Quantity to sample.
            settings (dict) : Transform settings used for sampling.
            random_state (int) : Seed for the random number generator.
            replace (bool) : Sample with or without replacement. Default value is False.

        Returns:
            LabelTimes : Random sample of labels.
        """
        sample = super().sample(random_state=random_state, replace=replace, **{key: value})

        if not self.settings.get('sample_in_transforms'):
            sample = sample.copy()
            sample.transforms.append(settings)

        return sample

    def _sample_per_label(self, key, value, settings, random_state=None, replace=False):
        """Returns a random sample per label.

        Args:
            key (str) : Determines the sampling method. Can either be 'n' or 'frac'.
            value (dict) : Quantity to sample per label.
            settings (dict) : Transform settings used for sampling.
            random_state (int) : Seed for the random number generator.
            replace (bool) : Sample with or without replacement. Default value is False.

        Returns:
            LabelTimes : Random sample per label.
        """
        self.settings['sample_in_transforms'] = True

        sample_per_label = []
        for label, value, in value.items():
            label = self[self[self.label_name] == label]
            sample = label._sample(key, value, settings, random_state=random_state, replace=replace)
            sample_per_label.append(sample)

        del self.settings['sample_in_transforms']
        sample = pd.concat(sample_per_label, axis=0, sort=False)
        sample = sample.copy()
        sample.transforms.append(settings)
        return sample

    def sample(self, n=None, frac=None, random_state=None, replace=False):
        """Return a random sample of labels.

        Args:
            n (int or dict) : Sample number of labels. A dictionary returns
                the number of samples to each label. Cannot be used with frac.
            frac (float or dict) : Sample fraction of labels. A dictionary returns
                the sample fraction to each label. Cannot be used with n.
            random_state (int) : Seed for the random number generator.
            replace (bool) : Sample with or without replacement. Default value is False.

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

            >>> labels.sample(n=3, random_state=0).sort_index()
              labels
            1      A
            2      B
            6      A

            Sample number per label:

            >>> n_per_label = {'A': 1, 'B': 2}
            >>> labels.sample(n=n_per_label, random_state=0).sort_index()
              labels
            3      B
            4      B
            5      A

            Sample fraction of labels:

            >>> labels.sample(frac=.4, random_state=2).sort_index()
              labels
            1      A
            3      B
            4      B

            Sample fraction per label:

            >>> frac_per_label = {'A': .5, 'B': .34}
            >>> labels.sample(frac=frac_per_label, random_state=2).sort_index()
              labels
            4      B
            5      A
            6      A
        """  # noqa
        settings = {
            'transform': 'sample',
            'n': n,
            'frac': frac,
            'random_state': random_state,
            'replace': replace,
        }

        key, value = ('n', n) if n else ('frac', frac)
        assert value, "must set value for 'n' or 'frac'"

        per_label = isinstance(value, dict)
        method = self._sample_per_label if per_label else self._sample
        sample = method(key, value, settings, random_state=random_state, replace=replace)
        return sample

    def infer_type(self):
        """Infer label type.

        Returns:
            str : Inferred label type. Either "continuous" or "discrete".
        """
        dtype = self[self.label_name].dtype
        is_discrete = pd.api.types.is_bool_dtype(dtype)
        is_discrete = is_discrete or pd.api.types.is_categorical_dtype(dtype)
        is_discrete = is_discrete or pd.api.types.is_object_dtype(dtype)

        if is_discrete:
            return 'discrete'

        return 'continuous'

    def equals(self, other, **kwargs):
        """Determines if two label time objects are the same.

        Args:
            other (LabelTimes) : Other label time object for comparison.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.equals method

        Returns:
            bool : Whether label time objects are the same.
        """
        return super().equals(other, **kwargs) and self.settings == other.settings

    def _load_settings(self, path):
        """Read the settings in json format from disk.

        Args:
            path (str) : Directory on disk to read from.
        """
        file = os.path.join(path, 'settings.json')
        assert os.path.exists(file), 'settings not found'

        with open(file, 'r') as file:
            settings = json.load(file)

        if 'dtypes' in settings:
            dtypes = settings.pop('dtypes')
            self = LabelTimes(self.astype(dtypes))

        self.settings.update(settings)

        return self

    def _save_settings(self, path):
        """Write the settings in json format to disk.

        Args:
            path (str) : Directory on disk to write to.
        """
        dtypes = self.dtypes.astype('str')
        self.settings['dtypes'] = dtypes.to_dict()

        file = os.path.join(path, 'settings.json')
        with open(file, 'w') as file:
            json.dump(self.settings, file)
            del self.settings['dtypes']

    def to_csv(self, path, filename='label_times.csv', save_settings=True, **kwargs):
        """Write label times in csv format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            filename (str) : Filename for label times. Default value is `label_times.csv`.
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_csv method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, filename)
        super().to_csv(file, **kwargs)

        if save_settings:
            self._save_settings(path)

    def to_parquet(self, path, filename='label_times.parquet', save_settings=True, **kwargs):
        """Write label times in parquet format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            filename (str) : Filename for label times. Default value is `label_times.parquet`.
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_parquet method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, filename)
        super().to_parquet(file, compression=None, engine='auto', **kwargs)

        if save_settings:
            self._save_settings(path)

    def to_pickle(self, path, filename='label_times.pickle', save_settings=True, **kwargs):
        """Write label times in pickle format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            filename (str) : Filename for label times. Default value is `label_times.pickle`.
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_pickle method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, filename)
        super().to_pickle(file, **kwargs)

        if save_settings:
            self._save_settings(path)
