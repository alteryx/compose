import numpy as np
import os
import pandas as pd


def _add_time_(df):
    df.reset_index(drop=True)
    df["order_time"] = np.nan

    columns = df.columns.tolist()
    days_since = columns.index("days_since_prior_order")
    hour_of_day = columns.index("order_hour_of_day")
    order_time = columns.index("order_time")

    df.iloc[0, order_time] = pd.Timestamp('Jan 1, 2015')
    df.iloc[0, order_time] += pd.Timedelta(df.iloc[0, hour_of_day], "h")

    for i in range(1, df.shape[0]):
        value = df.iloc[i - 1, order_time]
        value += pd.Timedelta(df.iloc[i, days_since], "d")
        value += pd.Timedelta(df.iloc[i, hour_of_day], "h")
        df.iloc[i, order_time] = value

    to_drop = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "eval_set"]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def load_orders(data_dir, nrows=None):
    path = os.path.join(data_dir, 'order_products__prior.csv')
    order_products = pd.read_csv(path, nrows=nrows)

    path = os.path.join(data_dir, 'orders.csv')
    orders = pd.read_csv(path, nrows=nrows)

    days = orders.days_since_prior_order
    days = days.dropna().astype('int').astype('str')
    days = days.add('d').apply(pd.Timedelta)

    path = os.path.join(data_dir, 'departments.csv')
    departments = pd.read_csv(path)

    path = os.path.join(data_dir, 'products.csv')
    products = pd.read_csv(path, nrows=nrows)

    df = order_products.merge(products).merge(departments)
    df = df.merge(orders).pipe(_add_time_)
    df.set_index('order_time', inplace=True)
    return df
