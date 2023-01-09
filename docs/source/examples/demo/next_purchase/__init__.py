import os
import pandas as pd
import requests
import tarfile
from demo import PWD, utils
from tqdm import tqdm

URL = r"https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz"
PWD = os.path.join(PWD, "next_purchase")


def _add_time(df, start="2015-01-01"):
    def timedelta(value, string):
        return pd.Timedelta(string.format(value))

    def process_orders(user):
        orders = user.groupby("order_id")
        days = orders.days_since_prior_order.first()
        days = days.cumsum().apply(timedelta, string="{}d")
        days = days.add(pd.Timestamp(start))
        hour_of_day = orders.order_hour_of_day.first()
        hour_of_day = hour_of_day.apply(timedelta, string="{}h")
        order_time = days.add(hour_of_day)
        return order_time

    df.days_since_prior_order.fillna(0, inplace=True)
    order_time = df.groupby("user_id").apply(process_orders).rename("order_time")

    columns = [
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
        "eval_set",
    ]

    df.drop(columns, axis=1, inplace=True)
    df = df.merge(order_time.reset_index(), on=["user_id", "order_id"])
    return df


def _data(nrows=1000000):
    output = os.path.join(PWD, "download")
    path = os.path.join(output, "instacart_2017_05_01")
    if not os.path.exists(path):
        utils.download(URL, output)

    file = os.path.join(path, "order_products__prior.csv")
    order_products = pd.read_csv(file, nrows=nrows)
    file = os.path.join(path, "products.csv")
    products = pd.read_csv(file)
    file = os.path.join(path, "departments.csv")
    departments = pd.read_csv(file)
    file = os.path.join(path, "orders.csv")
    orders = pd.read_csv(file, nrows=nrows)

    df = order_products.merge(products).merge(departments).merge(orders)
    df = df.pipe(_add_time)
    return df


def _read(file):
    path = os.path.join(PWD, file)
    df = pd.read_csv(path, parse_dates=["order_time"], index_col="id")
    return df


def load_sample():
    return _read("sample.csv")
