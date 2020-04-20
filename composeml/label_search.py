from composeml.utils import format_number, is_finite_number
from pandas import isnull


class ExampleSearch:
    def __init__(self, expected_count):
        self.expected_count = format_number(expected_count)
        self.reset_count()

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
        self.expected_count = sum(self.expected_label_counts.values())
        self.actual_label_counts = {}

    @property
    def is_complete(self):
        return all(map(self.is_label_complete, self.expected_label_counts))

    def is_label_complete(self, label):
        return self.actual_label_counts.get(label, 0) >= self.expected_label_counts[label]

    def is_label_incomplete(self, label):
        return self.actual_label_counts.get(label, 0) < self.expected_label_counts[label]

    def is_valid_labels(self, labels):
        not_null = super().is_valid_labels(labels)
        is_expected = not_null and all(label in self.expected_label_counts for label in labels.values())
        return is_expected and all(map(self.is_label_incomplete, labels.values()))

    def reset_count(self):
        self.actual_label_counts.clear()

    def update_count(self, labels):
        for label in labels.values():
            self.actual_label_counts[label] = self.actual_label_counts.get(label, 0) + 1
