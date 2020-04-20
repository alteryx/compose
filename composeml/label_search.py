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
    pass
