import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sns.set_context('paper')
sns.set_style('darkgrid')
COLOR = sns.color_palette("Set1", n_colors=100, desat=.75)


class LabelPlots:
    """Creates plots for Label Times."""

    def __init__(self, label_times):
        self._label_times = label_times

    def count_by_time(self, **kwargs):
        count_by_time = self._label_times.count_by_time

        if count_by_time is not None:
            fig, ax = plt.subplots()
            fig.autofmt_xdate()

            ax.stackplot(
                count_by_time.index,
                count_by_time.values.T,
                labels=count_by_time.columns,
                colors=COLOR,
                alpha=.9,
                **kwargs,
            )

            ax.legend(
                loc='upper left',
                title=self._label_times.name,
                facecolor='w',
                framealpha=.9,
            )

            ax.set_title('Label Count vs. Cutoff Time')
            ax.set_ylabel('Count')
            ax.set_xlabel('Time')
            return ax

    def distribution(self, **kwargs):
        dist = self._label_times[self._label_times.name]

        if self._label_times._is_categorical:
            ax = sns.countplot(dist, palette=COLOR, **kwargs)
        else:
            ax = sns.distplot(dist, kde=True, **kwargs)

        ax.set_title('Label Distribution')
        ax.set_ylabel('Count')
        return ax
