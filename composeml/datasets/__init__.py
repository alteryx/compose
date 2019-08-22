import pandas as pd
import os

DATA = os.path.join(os.path.dirname(__file__))


def transactions():
    path = os.path.join(DATA, 'transactions.csv')
    df = pd.read_csv(path, parse_dates=['transaction_time'])
    return df
