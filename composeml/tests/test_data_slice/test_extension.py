import pandas as pd

from composeml import LabelMaker


def test_context(transactions):
    lm = LabelMaker(target_entity='customer_id', time_index='time', window_size='1h')
    ds = next(lm.slice(transactions, num_examples_per_instance=1))

    assert isinstance(ds.context.customer_id, int)
    assert isinstance(ds.context.slice_number, int)
    assert isinstance(ds.context.slice_start, pd.Timestamp)
    assert isinstance(ds.context.slice_stop, pd.Timestamp)
    assert isinstance(ds.context.next_start, pd.Timestamp)
