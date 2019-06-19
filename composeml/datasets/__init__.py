from os import path

from pandas import read_csv


def _data():
    return path.join(path.dirname(__file__), 'data')


def transactions():
    file = path.join(_data(), 'transactions.csv')
    df = read_csv(file, parse_dates=['transaction_time'], index_col='transaction_time')
    return df
