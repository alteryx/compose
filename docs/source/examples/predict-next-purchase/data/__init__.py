import os
import tarfile

import pandas as pd
import requests

from tqdm import tqdm

PWD = os.path.dirname(__file__)
PATH = os.path.join(PWD, 'instacart_2017_05_01')


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


def load_orders(path=None, nrows=1000000):
    if not exists(): download()

    path = path or PATH

    file = os.path.join(path, 'order_products__prior.csv')
    order_products = pd.read_csv(file, nrows=nrows)

    file = os.path.join(path, 'orders.csv')
    orders = pd.read_csv(file, nrows=nrows)

    file = os.path.join(path, 'departments.csv')
    departments = pd.read_csv(file)

    file = os.path.join(path, 'products.csv')
    products = pd.read_csv(file)

    df = order_products.merge(products).merge(departments).merge(orders)
    df = df.pipe(add_time)
    return df


def exists():
    return os.path.exists(PATH)


def download():
    url = r'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
    response = requests.get(url, stream=True)
    assert response.status_code == 200, "unable to download data"

    bar_format = "Downloaded: {n}MB / {total}MB -{rate_fmt}, "
    bar_format += "Elapsed: {elapsed}, Remaining: {remaining}, Progress: {l_bar}{bar}"
    total = round(int(response.headers.get('content-length', 0)) / 1e+6)
    data = response.iter_content(chunk_size=int(1e+6))
    data = tqdm(data, total=total, unit="MB", bar_format=bar_format)

    tar = os.path.join(PWD, 'data.tar.gz')
    with open(tar, 'wb') as file:
        for chunk in data:
            file.write(chunk)

    with tarfile.open(tar, "r:gz") as file:
        file.extractall('data')

    os.remove(tar)
    response.close()


# def download():
#     assert not exists(), 'data already exists'

#     print('Downloading data..')
#     url = r'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
#     response = requests.get(url, stream=True)

#     if response.status_code == 200:
#         tar = os.path.join(PWD, 'data.tar.gz')

#         file = open(tar, 'wb')
#         file.write(response.raw.read())
#         file.close()

#         file = tarfile.open(tar, "r:gz")
#         file.extractall('data')
#         file.close()
#         os.remove(tar)

#     response.close()

#     if not exists():
#         raise FileNotFoundError('unable to download data')
