from abc import ABC, abstractmethod
from pandas import isnull


class LabelMakerSearch(ABC):
    @property
    @abstractmethod
    def total_examples(self):
        return NotImplementedError("This search must have `total_examples` as a property.")

    @property
    @abstractmethod
    def is_search_complete(self):
        return NotImplementedError("This search must have `is_search_complete` as a property.")

    @property
    @abstractmethod
    def is_finite_search(self):
        return NotImplementedError("This search must have `is_finite_search` as a property.")

    def is_label_valid(self, label):
        return NotImplementedError("This search must have `is_label_valid` as a method.")

    @abstractmethod
    def update_count(self, labels):
        return NotImplementedError("This search must have `is_label_valid` as a method.")

    @abstractmethod
    def reset_count(self):
        return NotImplementedError("This search must have `is_label_valid` as a method.")


class ExampleSearch(LabelMakerSearch):
    def __init__(self, expected_example_count):
        pass


class LabelSearch(LabelMakerSearch):
    def __init__(self, expected_label_counts):
        pass
