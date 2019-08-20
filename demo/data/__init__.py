import os
import pandas as pd


def add_time(df, start='2015-01-01'):
    def timedelta(value, string):
        return pd.Timedelta(string.format(value))

    def process_orders(user):
        orders = user.groupby('order_id')

        days = orders.days_since_prior_order.first()
        days = days.cumsum().apply(timedelta, string='{}d')
        days = days.add(pd.Timestamp(start))

        hour_of_day = orders.order_hour_of_day.first()
        hour_of_day = hour_of_day.apply(timedelta, string='{}h')

        order_time = days.add(hour_of_day)
        return order_time

    df.days_since_prior_order.fillna(0, inplace=True)
    order_time = df.groupby('user_id').apply(process_orders).rename('order_time')

    columns = [
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
        "eval_set",
    ]

    df.drop(columns, axis=1, inplace=True)
    df = df.merge(order_time.reset_index(), on=['user_id', 'order_id'])
    return df


def load_orders(data_dir, nrows={}):
    def path(key):
        file = '{}.csv'.format(key)
        value = os.path.join(data_dir, file)
        return value

    order_products = pd.read_csv(path('order_products__prior'), nrows=nrows)
    orders = pd.read_csv(path('orders'), nrows=nrows)
    departments = pd.read_csv(path('departments'))
    products = pd.read_csv(path('products'))

    df = order_products.merge(products).merge(departments).merge(orders)
    df = df.pipe(add_time)
    return df
