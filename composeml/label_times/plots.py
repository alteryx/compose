import matplotlib as mpl  # isort:skip
import pandas as pd
import seaborn as sns

# Raises an import error on OSX if not included.
# https://matplotlib.org/3.1.0/faq/osx_framework.html#working-with-matplotlib-on-osx
mpl.use("agg")  # noqa
pd.plotting.register_matplotlib_converters()
sns.set_context("notebook")
sns.set_style("darkgrid")
COLOR = sns.color_palette("Set1", n_colors=100, desat=0.75)


class LabelPlots:
    """Creates plots for Label Times."""

    def __init__(self, label_times):
        """Initializes Label Plots.

        Args:
            label_times (LabelTimes) : instance of Label Times
        """
        self._label_times = label_times

    def count_by_time(self, ax=None, **kwargs):
        """Plots the label distribution across cutoff times."""
        count_by_time = self._label_times.count_by_time
        count_by_time.sort_index(inplace=True)
        target_column = self._label_times.target_columns[0]

        ax = ax or mpl.pyplot.axes(label=id(self))
        vmin = count_by_time.index.min()
        vmax = count_by_time.index.max()
        ax.set_xlim(vmin, vmax)

        locator = mpl.dates.AutoDateLocator()
        formatter = mpl.dates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels():
            label.set_rotation(30)

        if len(count_by_time.shape) > 1:
            ax.stackplot(
                count_by_time.index,
                count_by_time.values.T,
                labels=count_by_time.columns,
                colors=COLOR,
                alpha=0.9,
                **kwargs,
            )

            ax.legend(
                loc="upper left",
                title=target_column,
                facecolor="w",
                framealpha=0.9,
            )

            ax.set_title("Label Count vs. Cutoff Times")
            ax.set_ylabel("Count")
            ax.set_xlabel("Time")

        else:
            ax.fill_between(
                count_by_time.index,
                count_by_time.values.T,
                color=COLOR[1],
            )

            ax.set_title("Label vs. Cutoff Times")
            ax.set_ylabel(target_column)
            ax.set_xlabel("Time")

        return ax

    @property
    def dist(self):
        """Alias for distribution."""
        return self.distribution

    def distribution(self, **kwargs):
        """Plots the label distribution."""
        self._label_times._assert_single_target()
        target_column = self._label_times.target_columns[0]
        dist = self._label_times[target_column]
        is_discrete = self._label_times.is_discrete[target_column]

        if is_discrete:
            ax = sns.countplot(x=dist, palette=COLOR, **kwargs)
        else:
            ax = sns.histplot(x=dist, kde=True, color=COLOR[1], **kwargs)

        ax.set_title("Label Distribution")
        ax.set_ylabel("Count")
        return ax
