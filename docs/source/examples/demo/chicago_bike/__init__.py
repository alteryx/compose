from demo import PWD
from pandas import read_csv
from os.path import join

PWD = join(PWD, 'chicago_bike')


def load_sample():
    return read_csv(
        join(PWD, 'sample.csv'),
        parse_dates=['starttime', 'stoptime'],
        index_col='trip_id',
    )
