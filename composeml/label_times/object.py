import json
import os

import pandas as pd

from ..version import __version__
from .description import describe_label_times
from .plots import LabelPlots

SCHEMA_VERSION = "0.1.0"


class LabelTimes(pd.DataFrame):
    """The data frame that contains labels and cutoff times for the target entity."""
    def __init__(
        self,
        data=None,
        target_entity=None,
        name=None,
        label_type=None,
        search_settings=None,
        transforms=None,
        *args,
        **kwargs,
    ):
        super().__init__(data=data, *args, **kwargs)
        self.target_entity = target_entity
        self.label_name = name
        self.label_type = label_type
        self.search_settings = search_settings or {}
        self.transforms = transforms or []
        self.plot = LabelPlots(self)

        if not self.empty:
            self._check_label_name()
            self._check_label_type()

    def _check_label_name(self):
        """Checks whether the target exists in the data frame."""
        if self.label_name is None:
            self.label_name = self._infer_label_name()

        info = 'target variable not found: %s' % self.label_name
        assert self.label_name in self.columns, info

    def _infer_label_name(self):
        """Infers the target name from the data frame.

        Returns:
            value (str): Inferred target name.
        """
        not_targets = [self.target_entity, 'time']
        target_names = self.columns.difference(not_targets)
        value = target_names.tolist()[0]
        return value

    def _check_label_type(self):
        """Checks whether the target type is continuous or discrete."""
        if self.label_type is None:
            self.label_type = self._infer_label_type()

        error = 'label type must be "continuous" or "discrete"'
        assert self.label_type in ['continuous', 'discrete'], error

    def _infer_label_type(self):
        """Infers the target type from the data type.

        Returns:
            value (str): Inferred label type. Either "continuous" or "discrete".
        """
        dtype = self[self.label_name].dtype
        is_discrete = pd.api.types.is_bool_dtype(dtype)
        is_discrete |= pd.api.types.is_categorical_dtype(dtype)
        is_discrete |= pd.api.types.is_object_dtype(dtype)
        value = 'discrete' if is_discrete else 'continuous'
        return value

    @property
    def settings(self):
        """Returns metadata about the label times."""
        return {
            'compose_version': __version__,
            'schema_version': SCHEMA_VERSION,
            'label_times': {
                'target_entity': self.target_entity,
                'label_name': self.label_name,
                'label_type': self.label_type,
                'search_settings': self.search_settings,
                'transforms': self.transforms,
            }
        }

    @property
    def is_discrete(self):
        """Whether labels are discrete."""
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
        """Prints out the settings used to make the label times."""
        describe_label_times(self)

    def copy(self, **kwargs):
        """Makes a copy of this object.

        Args:
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.copy method

        Returns:
            LabelTimes : Copy of label times.
        """
        label_times = super().copy(**kwargs)
        label_times.transforms = self.transforms.copy()
        return label_times

    def threshold(self, value, inplace=False):
        """Creates binary labels by testing if labels are above threshold.

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
        """Shifts the label times earlier for predicting in advance.

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
        """Bin labels into discrete intervals.

        Args:
            bins (int or array): The criteria to bin by.
                As an integer, the value can be the number of equal-width or quantile-based bins.
                If :code:`quantiles` is False, the value is defined as the number of equal-width bins.
                The range is extended by .1% on each side to include the minimum and maximum values.
                If :code:`quantiles` is True, the value is defined as the number of quantiles (e.g. 10 for deciles, 4 for quartiles, etc.)
                As an array, the value can be custom or quantile-based edges.
                If :code:`quantiles` is False, the value is defined as bin edges allowing for non-uniform width. No extension is done.
                If :code:`quantiles` is True, the value is defined as bin edges usings an array of quantiles (e.g. [0, .25, .5, .75, 1.] for quartiles)

            quantiles (bool): Determines whether to use a quantile-based discretization function.
            labels (array): Specifies the labels for the returned bins. Must be the same length as the resulting bins.
            right (bool) : Indicates whether bins includes the rightmost edge or not. Does not apply to quantile-based bins.

        Returns:
            LabelTimes : Instance of labels.

        Examples:
            These are the target values for the examples.

            >>> data = [226.93, 47.95, 283.46, 31.54]
            >>> lt = LabelTimes({'target': data})
            >>> lt
               target
            0  226.93
            1   47.95
            2  283.46
            3   31.54

            Bin values using equal-widths.

            >>> lt.bin(2)
                        target
            0  (157.5, 283.46]
            1  (31.288, 157.5]
            2  (157.5, 283.46]
            3  (31.288, 157.5]

            Bin values using custom-widths.

            >>> lt.bin([0, 200, 400])
                   target
            0  (200, 400]
            1    (0, 200]
            2  (200, 400]
            3    (0, 200]

            Bin values using quartiles.

            >>> lt.bin(4, quantiles=True)
                                     target
            0             (137.44, 241.062]
            1              (43.848, 137.44]
            2             (241.062, 283.46]
            3  (31.538999999999998, 43.848]

            Assign labels to bins.

            >>> lt.bin(2, labels=['low', 'high'])
              target
            0   high
            1    low
            2   high
            3    low
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
        sample_per_label = []
        for label, value, in value.items():
            label = self[self[self.label_name] == label]
            sample = label._sample(key, value, settings, random_state=random_state, replace=replace)
            sample_per_label.append(sample)

        sample = pd.concat(sample_per_label, axis=0, sort=False)
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
            These are the label values for the examples.

            >>> lt = LabelTimes({'labels': list('AABBBAA')})
            >>> lt
              labels
            0      A
            1      A
            2      B
            3      B
            4      B
            5      A
            6      A

            Sample a number of examples.

            >>> lt.sample(n=3, random_state=0)
              labels
            1      A
            2      B
            6      A

            Sample a number of examples for specific labels.

            >>> n_per_label = {'A': 1, 'B': 2}
            >>> lt.sample(n=n_per_label, random_state=0)
              labels
            3      B
            4      B
            5      A

            Sample a fraction of the examples.

            >>> lt.sample(frac=.4, random_state=2)
              labels
            1      A
            3      B
            4      B

            Sample a fraction of the examples for specific labels.

            >>> frac_per_label = {'A': .5, 'B': .34}
            >>> lt.sample(frac=frac_per_label, random_state=2)
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

        sample = sample.copy()
        sample.sort_index(inplace=True)
        sample.transforms.append(settings)
        return sample

    def equals(self, other, **kwargs):
        """Determines if two label time objects are the same.

        Args:
            other (LabelTimes) : Other label time object for comparison.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.equals method

        Returns:
            bool : Whether label time objects are the same.
        """
        is_equal = super().equals(other, **kwargs)
        is_equal &= self.settings == other.settings
        return is_equal

    def _load_settings(self, path):
        """Read the settings in json format from disk.

        Args:
            path (str) : Directory on disk to read from.
        """
        file = os.path.join(path, 'settings.json')
        assert os.path.exists(file), 'settings not found'

        with open(file, 'r') as file:
            settings = json.load(file)

        df = self
        if 'dtypes' in settings:
            dtypes = settings.pop('dtypes')
            df = df.astype(dtypes)

        kwargs = settings['label_times']
        name = kwargs.pop('label_name')
        self = LabelTimes(data=df, name=name, **kwargs)
        return self

    def _save_settings(self, path):
        """Write the settings in json format to disk.

        Args:
            path (str) : Directory on disk to write to.
        """
        settings = self.settings
        dtypes = self.dtypes.astype('str')
        settings['dtypes'] = dtypes.to_dict()

        file = os.path.join(path, 'settings.json')
        with open(file, 'w') as file:
            json.dump(settings, file)

    def to_csv(self, path, save_settings=True, **kwargs):
        """Write label times in csv format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_csv method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, 'data.csv')
        super().to_csv(file, index=False, **kwargs)

        if save_settings:
            self._save_settings(path)

    def to_parquet(self, path, save_settings=True, **kwargs):
        """Write label times in parquet format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_parquet method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, 'data.parquet')
        super().to_parquet(file, compression=None, engine='auto', **kwargs)

        if save_settings:
            self._save_settings(path)

    def to_pickle(self, path, save_settings=True, **kwargs):
        """Write label times in pickle format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_pickle method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, 'data.pickle')
        super().to_pickle(file, **kwargs)

        if save_settings:
            self._save_settings(path)

    # ----------------------------------------
    # Subclassing Pandas Data Frame
    # ----------------------------------------

    _metadata = [
        '_recursive',
        'label_name',
        'label_type',
        'search_settings',
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
        return LabelTimes
