import numpy as np
import os
import pandas as pd


def add_time(df):
    def apply(user):
        user.sort_values('order_id', inplace=True)
        user.days_since_prior_order.fillna(1, inplace=True)
        orders = user.groupby('order_id')

        prior_order = orders.days_since_prior_order.first()
        prior_order = prior_order.cumsum().apply('{:.0f}d'.format)
        prior_order = prior_order.apply(pd.Timedelta)
        prior_order = prior_order.add(pd.Timestamp('2010-01-01'))

        return prior_order

    return df.groupby('user_id').apply(apply)


def add_time(df):
    df.reset_index(drop=True)
    df["order_time"] = np.nan

    df.days_since_prior_order = df.days_since_prior_order.fillna(0)
    df.order_hour_of_day = df.order_hour_of_day.fillna(0)

    days_since = df.columns.tolist().index("days_since_prior_order")
    hour_of_day = df.columns.tolist().index("order_hour_of_day")
    order_time = df.columns.tolist().index("order_time")

    df.iloc[0, order_time] = pd.Timestamp('Jan 1, 2015') + pd.Timedelta(df.iloc[0, hour_of_day], "h")
    for i in range(1, df.shape[0]):
        df.iloc[i, order_time] = df.iloc[i - 1, order_time] \
            + pd.Timedelta(df.iloc[i, days_since], "d") \
            + pd.Timedelta(df.iloc[i, hour_of_day], "h")

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
    products = pd.read_csv(path)

    df = order_products.merge(products).merge(departments)
    df = df.merge(orders)  # .pipe(add_time)
    # df.set_index('order_time', inplace=True)
    return df
