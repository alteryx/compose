import os
import pandas as pd
from demo import utils

URL = r"https://ti.arc.nasa.gov/c/6/"
PWD = os.path.dirname(__file__)


def _download_data():
    output = os.path.join(PWD, "download")
    utils.download(URL, output)


def _data():
    path = os.path.join(PWD, "download", "train_FD004.txt")
    if not os.path.exists(path):
        _download_data()
    cols = ["engine_no", "time_in_cycles"]
    cols += ["operational_setting_{}".format(i + 1) for i in range(3)]
    cols += ["sensor_measurement_{}".format(i + 1) for i in range(26)]
    df = pd.read_csv(path, sep=" ", header=None, names=cols)
    df = df.drop(cols[-5:], axis=1).rename_axis("id")
    df["time"] = pd.date_range("1/1/2000", periods=df.shape[0], freq="600s")
    return df


def _read(file):
    path = os.path.join(PWD, file)
    df = pd.read_csv(path, parse_dates=["time"], index_col="id")
    return df


def load_sample():
    return _read("sample.csv")
