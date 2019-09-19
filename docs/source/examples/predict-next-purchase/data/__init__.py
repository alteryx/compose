import os
import pandas as pd
import requests
import tarfile

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
    assert not exists(), 'data already exists'

    print('Downloading data..')
    url = r'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz?X-Amz-Expires=21600&X-Amz-Date=20190919T151856Z&X-Amz-Security-Token=FQoGZXIvYXdzEO7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGD6aHa8GhLQmH%2BSEiKVBO81XbnVwOEHjb14XwjJxrnQ%2B9AdZW%2FBb8xxCTwF%2BZpvyQHY%2F1UF8B8Ati9oNTUvo7IhH%2FyUZ4QCWqzzQQBgH7yYn3a2saCbuA%2Bajhm8z%2FdYZzYM%2FzBQStCgoLdaprCL7ya1DzH0megTogInsZOp4JaoynZTnr0kPxeaRpRgQX3H9ScaDrNUVA1N4fg4KEHuTlWK%2Fv2%2BMUF0sMnwANVJKpGNYIAK%2BMaG4gm3wktFLraDBVNfNdqoNkWwVlGU6loT273hqalUnEBi55WY5Ao3IbWGb38marL3t6cyzkAjUQp8t20upnoUFj3wuywOs3fImUqgTYvz%2B1QPJzMEPykluPMD5J4NWp%2Bo3d9chlIxAT2x83xQI%2FUpTKwndUxDu4o1nH1K%2B7SyltbUAxeQ3VFp0xWo0Rxlufty9eYorL5blSUkTyml41Xr6xz0KE5KJgLOf94yx6XKQj8OjhXAs9iBKo9ldbZziwmZnYLHAAAhh33uJjlMSuvX28bLk75oTUne6sZhfEm8UiU3ngXKQ8xX%2FXFEEMZPu7uEWrPQPSTtWzwb8uIhksasGIjzGKP86kEZSEgWc%2FY64g9%2B8J4lKapYCxmM2QNcuu7pjEMcycL%2FxDJFH6z1WDhdYQSLfKOTTFhSesd%2FUcXJBgSDKPxTAYgeR2v8WKqJuIv0DDuyGzbZV3yWUMcOV6Q1UDSceYvxb3aetZ3fxSSQKLnvjewF&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASDAVXBX7SLZST3EZ%2F20190919%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Signature=f69bf493562f36add7efcadaf8e98ae10cf3d8eb8ff249f858fb7091ecc057d3'
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        tar = os.path.join(PWD, 'data.tar.gz')

        file = open(tar, 'wb')
        file.write(response.raw.read())
        file.close()

        file = tarfile.open(tar, "r:gz")
        file.extractall('data')
        file.close()
        os.remove(tar)

    response.close()

    if not exists():
        raise FileNotFoundError('unable to download data')
