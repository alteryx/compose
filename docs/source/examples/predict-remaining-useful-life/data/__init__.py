import pandas as pd


def load(path):
    operational_settings = ['operational_setting_{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sensor_measurement_{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns
    data = pd.read_csv(path, sep=' ', header=None, names=cols)
    data = data.drop(cols[-5:], axis=1)
    data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')
    return data
