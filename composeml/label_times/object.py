import json
import os

import pandas as pd

from ..version import __version__
from .description import describe_label_times
from .plots import LabelPlots

SCHEMA_VERSION = "0.1.0"


class LabelTimes(pd.DataFrame):
    """The data frame that contains labels and cutoff times for the target dataframe."""

    def __init__(
        self,
        data=None,
        target_dataframe_name=None,
        target_types=None,
        target_columns=None,
        search_settings=None,
        transforms=None,
        *args,
        **kwargs,
    ):
        super().__init__(data=data, *args, **kwargs)
        self.target_dataframe_name = target_dataframe_name
        self.target_columns = target_columns or []
        self.target_types = target_types or {}
        self.search_settings = search_settings or {}
        self.transforms = transforms or []
        self.plot = LabelPlots(self)

        if not self.empty:
            self._check_label_times()

    def _assert_single_target(self):
        """Asserts that the label times object contains a single target."""
        info = "must first select an individual target"
        assert self._is_single_target, info

    def _check_target_columns(self):
        """Validates the target columns."""
        if not self.target_columns:
            self.target_columns = self._infer_target_columns()
        else:
            for target in self.target_columns:
                info = 'target "%s" not found in data frame'
                assert target in self.columns, info % target

    def _check_target_types(self):
        """Validates the target types."""
        if isinstance(self.target_types, dict):
            self.target_types = pd.Series(self.target_types, dtype="object")

        if self.target_types.empty:
            self.target_types = self._infer_target_types()
        else:
            target_names = self.target_types.index.tolist()
            match = target_names == self.target_columns
            assert match, "target names in types must match target columns"

    def _check_label_times(self):
        """Validates the lables times object."""
        self._check_target_columns()
        self._check_target_types()

    def _infer_target_columns(self):
        """Infers the names of the targets in the data frame.

        Returns:
            value (list): A list of the target names.
        """
        not_targets = [self.target_dataframe_name, "time"]
        target_columns = self.columns.difference(not_targets)
        assert not target_columns.empty, "target columns not found"
        value = target_columns.tolist()
        return value

    @property
    def _is_single_target(self):
        return len(self.target_columns) == 1

    def _get_target_type(self, dtype):
        is_discrete = pd.api.types.is_bool_dtype(dtype)
        is_discrete |= pd.api.types.is_categorical_dtype(dtype)
        is_discrete |= pd.api.types.is_object_dtype(dtype)
        value = "discrete" if is_discrete else "continuous"
        return value

    def _infer_target_types(self):
        """Infers the target type from the data type.

        Returns:
            types (Series): Inferred label type. Either "continuous" or "discrete".
        """
        dtypes = self.dtypes[self.target_columns]
        types = dtypes.apply(self._get_target_type)
        return types

    def select(self, target):
        """Selects one of the target variables.

        Args:
            target (str): The name of the target column.

        Returns:
            lt (LabelTimes): A label times object that contains a single target.

        Examples:
            Create a label times object that contains multiple target variables.

            >>> entity = [0, 0, 1, 1]
            >>> labels = [True, False, True, False]
            >>> time = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']
            >>> data = {'entity': entity, 'time': time, 'A': labels, 'B': labels}
            >>> lt = LabelTimes(data=data, target_dataframe_name='entity', target_columns=['A', 'B'])
            >>> lt
               entity        time      A      B
            0       0  2020-01-01   True   True
            1       0  2020-01-02  False  False
            2       1  2020-01-03   True   True
            3       1  2020-01-04  False  False

            Select a single target from the label times.

            >>> lt.select('B')
               entity        time      B
            0       0  2020-01-01   True
            1       0  2020-01-02  False
            2       1  2020-01-03   True
            3       1  2020-01-04  False
        """
        assert not self._is_single_target, "only one target exists"
        if not isinstance(target, str):
            raise TypeError("target name must be string")
        assert target in self.target_columns, 'target "%s" not found' % target

        lt = self.copy()
        lt.target_columns = [target]
        lt.target_types = lt.target_types[[target]]
        lt = lt[[self.target_dataframe_name, "time", target]]
        return lt

    @property
    def settings(self):
        """Returns metadata about the label times."""
        return {
            "compose_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "label_times": {
                "target_dataframe_name": self.target_dataframe_name,
                "target_columns": self.target_columns,
                "target_types": self.target_types.to_dict(),
                "search_settings": self.search_settings,
                "transforms": self.transforms,
            },
        }

    @property
    def is_discrete(self):
        """Whether labels are discrete."""
        return self.target_types.eq("discrete")

    @property
    def distribution(self):
        """Returns label distribution if labels are discrete."""
        self._assert_single_target()
        target_column = self.target_columns[0]

        if self.is_discrete[target_column]:
            labels = self.assign(count=1)
            labels = labels.groupby(target_column)
            distribution = labels["count"].count()
            return distribution
        else:
            return self[target_column].describe()

    @property
    def count(self):
        """Returns label count per instance."""
        self._assert_single_target()
        count = self.groupby(self.target_dataframe_name)
        count = count[self.target_columns[0]].count()
        count = count.to_frame("count")
        return count

    @property
    def count_by_time(self):
        """Returns label count across cutoff times."""
        self._assert_single_target()
        target_column = self.target_columns[0]

        if self.is_discrete[target_column]:
            keys = ["time", target_column]
            value = self.groupby(keys).time.count()
            value = value.unstack(target_column).fillna(0)
        else:
            value = self.groupby("time")
            value = value[target_column].count()

        value = (
            value.cumsum()
        )  # In Python 3.5, these values automatically convert to float.
        value = value.astype("int")
        return value

    def describe(self):
        """Prints out the settings used to make the label times."""
        if not self.empty:
            self._assert_single_target()
            describe_label_times(self)

    def copy(self, deep=True):
        """Make a copy of this object's indices and data.

        Args:
            deep (bool): Make a deep copy, including a copy of the data and the indices.
                With ``deep=False`` neither the indices nor the data are copied. Default is True.

        Returns:
            lt (LabelTimes): A copy of the label times object.
        """
        lt = super().copy(deep=deep)
        lt.target_dataframe_name = self.target_dataframe_name
        lt.target_columns = self.target_columns
        lt.target_types = self.target_types.copy()
        lt.search_settings = self.search_settings.copy()
        lt.transforms = self.transforms.copy()
        return lt

    def threshold(self, value, inplace=False):
        """Creates binary labels by testing if labels are above threshold.

        Args:
            value (float) : Value of threshold.
            inplace (bool) : Modify labels in place.

        Returns:
            labels (LabelTimes) : Instance of labels.
        """
        self._assert_single_target()
        target_column = self.target_columns[0]
        labels = self if inplace else self.copy()
        labels[target_column] = labels[target_column].gt(value)
        labels.target_types[target_column] = "discrete"

        transform = {"transform": "threshold", "value": value}
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
        labels["time"] = labels["time"].sub(pd.Timedelta(value))

        transform = {"transform": "apply_lead", "value": value}
        labels.transforms.append(transform)

        if not inplace:
            return labels

    def bin(self, bins, quantiles=False, labels=None, right=True, precision=3):
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
            precision (int): The precision at which to store and display the bins labels. Default value is 3.

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

            Bin values using infinite edges.

            >>> lt.bin(['-inf', 100, 'inf'])
                      target
            0   (100.0, inf]
            1  (-inf, 100.0]
            2   (100.0, inf]
            3  (-inf, 100.0]

            Bin values using quartiles.

            >>> lt.bin(4, quantiles=True)
                                     target
            0             (137.44, 241.062]
            1              (43.848, 137.44]
            2             (241.062, 283.46]
            3  (31.538999999999998, 43.848]

            Bin values using custom quantiles with precision.

            >>> lt.bin([0, .5, 1], quantiles=True, precision=1)
                       target
            0  (137.4, 283.5]
            1   (31.4, 137.4]
            2  (137.4, 283.5]
            3   (31.4, 137.4]

            Assign labels to bins.

            >>> lt.bin(2, labels=['low', 'high'])
              target
            0   high
            1    low
            2   high
            3    low
        """  # noqa
        self._assert_single_target()
        target_column = self.target_columns[0]
        values = self[target_column].values

        if quantiles:
            values = pd.qcut(values, q=bins, labels=labels, precision=precision)

        else:
            if isinstance(bins, list):
                for i, edge in enumerate(bins):
                    if edge in ["-inf", "inf"]:
                        bins[i] = float(edge)

            values = pd.cut(
                values, bins=bins, labels=labels, right=right, precision=precision
            )

        transform = {
            "transform": "bin",
            "bins": bins,
            "quantiles": quantiles,
            "labels": labels,
            "right": right,
            "precision": precision,
        }

        lt = self.copy()
        lt[target_column] = values
        lt.transforms.append(transform)
        lt.target_types[target_column] = "discrete"
        return lt

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
        sample = super().sample(
            random_state=random_state, replace=replace, **{key: value}
        )
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
        target_column = self.target_columns[0]

        for (
            label,
            value,
        ) in value.items():
            label = self[self[target_column] == label]
            sample = label._sample(
                key, value, settings, random_state=random_state, replace=replace
            )
            sample_per_label.append(sample)

        sample = pd.concat(sample_per_label, axis=0, sort=False)
        return sample

    def sample(
        self, n=None, frac=None, random_state=None, replace=False, per_instance=False
    ):
        """Return a random sample of labels.

        Args:
            n (int or dict) : Sample number of labels. A dictionary returns
                the number of samples to each label. Cannot be used with frac.
            frac (float or dict) : Sample fraction of labels. A dictionary returns
                the sample fraction to each label. Cannot be used with n.
            random_state (int) : Seed for the random number generator.
            replace (bool) : Sample with or without replacement. Default value is False.
            per_instance (bool): Whether to apply sampling to each group. Default is False.

        Returns:
            LabelTimes : Random sample of labels.

        Examples:
            Create a label times object.

            >>> entity = [0, 0, 1, 1]
            >>> labels = [True, False, True, False]
            >>> data = {'entity': entity, 'labels': labels}
            >>> lt = LabelTimes(data=data, target_dataframe_name='entity', target_columns=['labels'])
            >>> lt
               entity  labels
            0       0    True
            1       0   False
            2       1    True
            3       1   False

            Sample a number of the examples.

            >>> lt.sample(n=3, random_state=0)
               entity  labels
            1       0   False
            2       1    True
            3       1   False

            Sample a fraction of the examples.

            >>> lt.sample(frac=.25, random_state=0)
               entity  labels
            2       1    True

            Sample a number of the examples for specific labels.

            >>> n = {True: 1, False: 1}
            >>> lt.sample(n=n, random_state=0)
               entity  labels
            2       1    True
            3       1   False

            Sample a fraction of the examples for specific labels.

            >>> frac = {True: .5, False: .5}
            >>> lt.sample(frac=frac, random_state=0)
               entity  labels
            2       1    True
            3       1   False

            Sample a number of the examples from each entity group.

            >>> lt.sample(n={True: 1}, per_instance=True, random_state=0)
               entity  labels
            0       0    True
            2       1    True

            Sample a fraction of the examples from each entity group.

            >>> lt.sample(frac=.5, per_instance=True, random_state=0)
               entity  labels
            1       0   False
            3       1   False
        """  # noqa
        self._assert_single_target()

        settings = {
            "transform": "sample",
            "n": n,
            "frac": frac,
            "random_state": random_state,
            "replace": replace,
            "per_instance": per_instance,
        }

        key, value = ("n", n) if n else ("frac", frac)
        assert value, "must set value for 'n' or 'frac'"

        per_label = isinstance(value, dict)
        method = "_sample_per_label" if per_label else "_sample"

        def transform(lt):
            sample = getattr(lt, method)(
                key=key,
                value=value,
                settings=settings,
                random_state=random_state,
                replace=replace,
            )
            return sample

        if per_instance:
            groupby = self.groupby(self.target_dataframe_name, group_keys=False)
            sample = groupby.apply(transform)
        else:
            sample = transform(self)

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

    def _save_settings(self, path):
        """Write the settings in json format to disk.

        Args:
            path (str) : Directory on disk to write to.
        """
        settings = self.settings
        dtypes = self.dtypes.astype("str")
        settings["dtypes"] = dtypes.to_dict()

        file = os.path.join(path, "settings.json")
        with open(file, "w") as file:
            json.dump(settings, file)

    def to_csv(self, path, save_settings=True, **kwargs):
        """Write label times in csv format to disk.

        Args:
            path (str) : Location on disk to write to (will be created as a directory).
            save_settings (bool) : Whether to save the settings used to make the label times.
            **kwargs: Keyword arguments to pass to underlying pandas.DataFrame.to_csv method
        """
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, "data.csv")
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
        file = os.path.join(path, "data.parquet")
        super().to_parquet(file, compression=None, engine="auto", **kwargs)

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
        file = os.path.join(path, "data.pickle")
        super().to_pickle(file, **kwargs)

        if save_settings:
            self._save_settings(path)

    # ----------------------------------------
    # Subclassing Pandas Data Frame
    # ----------------------------------------

    _metadata = [
        "search_settings",
        "target_columns",
        "target_dataframe_name",
        "target_types",
        "transforms",
    ]

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other label times data frames.

        Args:
            other (LabelTimes) : The label times from which to get the attributes from.
            method (str) : A passed method name for optionally taking different types of propagation actions based on this value.
        """
        if method == "concat":
            other = other.objs[0]

            for key in self._metadata:
                value = getattr(other, key, None)
                setattr(self, key, value)

            return self

        return super().__finalize__(other=other, method=method, **kwargs)

    @property
    def _constructor(self):
        return LabelTimes
