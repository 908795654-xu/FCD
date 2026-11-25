import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


def _build_time_features(df_stamp, timeenc, freq):
    df_stamp = df_stamp.copy()
    df_stamp['date'] = pd.to_datetime(df_stamp['date'])
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        if freq == 't':
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        return df_stamp.drop(['date'], axis=1).values
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
    return data_stamp.transpose(1, 0)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', scaler_data_path=None, target_data_path=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_data_path = scaler_data_path
        self.target_data_path = target_data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        target_path = self.target_data_path or self.data_path
        df_target_raw = df_raw if target_path == self.data_path else pd.read_csv(os.path.join(self.root_path, target_path))
        scaler_path = self.scaler_data_path or self.data_path
        if scaler_path != self.data_path:
            df_scaler = pd.read_csv(os.path.join(self.root_path, scaler_path))
        else:
            df_scaler = df_raw

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
            df_target_data = df_target_raw[df_target_raw.columns[1:]]
            df_scaler_data = df_scaler[df_scaler.columns[1:]]
        else:
            df_data = df_raw[[self.target]]
            df_target_data = df_target_raw[[self.target]]
            df_scaler_data = df_scaler[[self.target]]

        if self.scale:
            self.scaler.fit(df_scaler_data[border1s[0]:border2s[0]].values)
            data_x = self.scaler.transform(df_data.values)
            data_y = self.scaler.transform(df_target_data.values)
        else:
            data_x = df_data.values
            data_y = df_target_data.values

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = _build_time_features(df_target_raw[['date']][border1:border2], self.timeenc, self.freq)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', scaler_data_path=None, target_data_path=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_data_path = scaler_data_path
        self.target_data_path = target_data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        target_path = self.target_data_path or self.data_path
        df_target_raw = df_raw if target_path == self.data_path else pd.read_csv(os.path.join(self.root_path, target_path))
        scaler_path = self.scaler_data_path or self.data_path
        df_scaler_raw = df_raw if scaler_path == self.data_path else pd.read_csv(os.path.join(self.root_path, scaler_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
            df_target_data = df_target_raw[df_target_raw.columns[1:]]
            df_scaler_data = df_scaler_raw[df_scaler_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]
            df_target_data = df_target_raw[[self.target]]
            df_scaler_data = df_scaler_raw[[self.target]]

        if self.scale:
            # Fit scaler on the (optional) provided scaler dataset, but apply it to the current data_path.
            self.scaler.fit(df_scaler_data[border1s[0]:border2s[0]].values)
            data_x = self.scaler.transform(df_data.values)
            data_y = self.scaler.transform(df_target_data.values)
        else:
            data_x = df_data.values
            data_y = df_target_data.values

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = _build_time_features(df_target_raw[['date']][border1:border2], self.timeenc, self.freq)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', scaler_data_path=None, target_data_path=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_data_path = scaler_data_path
        self.target_data_path = target_data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        target_path = self.target_data_path or self.data_path
        df_target_raw = df_raw if target_path == self.data_path else pd.read_csv(os.path.join(self.root_path, target_path))
        scaler_path = self.scaler_data_path or self.data_path
        if scaler_path != self.data_path:
            df_scaler = pd.read_csv(os.path.join(self.root_path, scaler_path))
        else:
            df_scaler = df_raw
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        df_target_raw = df_target_raw[['date'] + cols + [self.target]]

        n_total = len(df_raw)
        num_train = int(n_total * 0.7)
        num_test = int(n_total * 0.2)
        num_vali = n_total - num_train - num_test
        border1s = [0, num_train - self.seq_len, n_total - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, n_total]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
            df_target_data = df_target_raw[df_target_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]
            df_target_data = df_target_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data_x = self.scaler.transform(df_data.values)
            data_y = self.scaler.transform(df_target_data.values)
        else:
            data_x = df_data.values
            data_y = df_target_data.values

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = _build_time_features(df_target_raw[['date']][border1:border2], self.timeenc, self.freq)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
