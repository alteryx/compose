from pandas import isnull


class ExampleSearch:
    def __init__(self, expected_example_count):
        pass

    @property
    def total_examples(self):
        pass

    @property
    def is_search_complete(self):
        pass

    @property
    def is_finite_search(self):
        pass

    def is_label_valid(self, label):
        return not isnull(label)

    def update_count(self, labels):
        pass

    def reset_count(self):
        pass


class LabelSearch(ExampleSearch):
    pass
