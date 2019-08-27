import seaborn as sns

sns.set_context('paper')
sns.set_style('whitegrid', {'grid.color': '.9'})
COLOR = sns.color_palette("Set1", n_colors=100, desat=.75)


class Plots:
    """Plots for Label Times."""

    def __init__(self, label_times):
        self._label_times = label_times

    def count_by_time(self):
        count = self._label_times.count_by_time
        plot = count.plot.area(color=COLOR, alpha=.9)
        return plot

    def distribution(self):
        if self._label_times._is_categorical(frac=.1, thresh=.9):
            dist = self._label_times.distribution
            plot = dist.plot(kind='bar', color=COLOR, alpha=.9)
            plot.set_title('Label Distribution')
            plot.set_ylabel('Count')

        else:
            dist = self._label_times[self._label_times.name]
            plot = sns.distplot(dist, kde=True, alpha=.9)

        return plot
