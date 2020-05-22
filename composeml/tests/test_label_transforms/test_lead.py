import pandas as pd


def test_lead(labels):
    labels = labels.apply_lead('10min')
    transform = labels.transforms[0]

    assert transform['transform'] == 'apply_lead'
    assert transform['value'] == '10min'

    answer = [
        '2014-01-01 00:35:00',
        '2014-01-01 00:38:00',
        '2013-12-31 23:51:00',
        '2013-12-31 23:54:00',
    ]

    time = pd.Series(answer, name='time', dtype='datetime64[ns]')
    time = time.rename_axis('label_id')

    pd.testing.assert_series_equal(labels['time'], time)
