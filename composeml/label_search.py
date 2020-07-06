from collections import Counter

from pandas import isnull


class ExampleSearch:
    """A label search based on the number of examples.

    Args:
        expected_count (int): The expected number of examples to find.
    """

    def __init__(self, expected_count):
        self.expected_count = self._check_number(expected_count)
        self.reset_count()

    @staticmethod
    def _check_number(n):
        """Checks and formats the expected number of examples."""
        if n == -1 or n == 'inf':
            return float('inf')
        else:
            info = 'expected count must be numeric'
            assert isinstance(n, (int, float)), info
            return n

    @staticmethod
    def _is_finite_number(n):
        """Checks if a number if finite."""
        return n > 0 and abs(n) != float('inf')

    @property
    def is_complete(self):
        """Whether the search has found the expected number of examples."""
        return self.actual_count >= self.expected_count

    @property
    def is_finite(self):
        """Whether the expected number of examples is a finite number."""
        return self._is_finite_number(self.expected_count)

    def is_valid_labels(self, labels):
        """Whether the label values are not null."""
        return not any(map(isnull, labels.values()))

    def reset_count(self):
        """Reset the internal count of actual labels."""
        self.actual_count = 0

    def update_count(self, labels):
        """Update the internal count of actual labels."""
        self.actual_count += 1


class LabelSearch(ExampleSearch):
    """A label search based on the number of examples for each label.

    Args:
        expected_label_counts (dict): The expected number of examples to be find for each label.
            The dictionary should map a label to the number of examples to find for the label.
    """

    def __init__(self, expected_label_counts):
        items = expected_label_counts.items()
        self.expected_label_counts = Counter({label: self._check_number(count) for label, count in items})
        self.expected_count = sum(self.expected_label_counts.values())
        self.actual_label_counts = Counter()

    @property
    def is_complete(self):
        """Whether the search has found the expected number of examples for each label."""
        return len(self.expected_label_counts - self.actual_label_counts) == 0

    def is_complete_label(self, label):
        """Whether the search has found the expected number of examples for a label."""
        return self.actual_label_counts.get(label, 0) >= self.expected_label_counts[label]

    def is_valid_labels(self, labels):
        """Whether label values meet the search criteria.

        The search criteria is defined as label values that are not null, expected by the user, and have not reached the expected count.
        When these conditions are met by any label value, the labels are set to return to the user.
        This includes the other label values which share the same cutoff time.

        Args:
            labels (dict): The actual label values found during a search.

        Returns:
            value (bool): The value is True when valid, otherwise False.
        """
        label_values = labels.values()
        not_null = super().is_valid_labels(labels)
        is_expected = not_null and any(label in self.expected_label_counts for label in label_values)
        value = is_expected and any(not self.is_complete_label(label) for label in label_values)
        return value

    def reset_count(self):
        """Reset the internal count of actual labels."""
        self.actual_label_counts.clear()

    def update_count(self, labels):
        """Update the internal count of the actual labels.

        Args:
            labels (dict): The actual label values found during a search.
        """
        self.actual_label_counts.update(labels.values())
