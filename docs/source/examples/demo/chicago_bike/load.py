from demo import PWD
from pandas import read_csv
from os.path import join

PWD = join(PWD, 'chicago_bike')


def _read(file):
    return read_csv(
        join(PWD, file),
        parse_dates=['starttime', 'stoptime'],
        index_col='trip_id',
    )


def historical_sample():
    return _read('historical.csv')


def future_sample():
    return _read('future.csv')
