from composeml.utils import format_number, is_finite_number
from pandas import isnull


class ExampleSearch:
    def __init__(self, expected_count):
        self.expected_count = format_number(expected_count)
        self.reset_count()

    @property
    def expected_count(self):
        return self.expected_count

    @property
    def is_complete(self):
        return self.actual_count >= self.expected_count

    @property
    def is_finite(self):
        return is_finite_number(self.expected_count)

    def is_valid_labels(self, labels):
        return not any(map(isnull, labels.values()))

    def update_count(self, labels):
        self.actual_count += 1

    def reset_count(self):
        self.actual_count = 0


class LabelSearch(ExampleSearch):
    def __init__(self, expected_label_counts):
        items = expected_label_counts.items()
        self.expected_label_counts = {label: format_number(count) for label, count in items}
        self.example_count = sum(self.expected_label_counts.values())
        self.actual_label_counts = {}

    @property
    def is_complete(self):
        items = self.expected_label_counts.items()
        return all(self.actual_label_counts.get(label, 0) >= count for label, count in items)

    def is_valid_labels(self, labels):
        return super().is_valid_labels(labels) and \
            all(label in self.expected_label_counts for label in labels.values())

    def reset_count(self):
        self.actual_label_counts.clear()

    def update_count(self, labels):
        for label in labels.values():
            self.actual_label_counts[label] = self.actual_label_counts.get(label, 0) + 1
