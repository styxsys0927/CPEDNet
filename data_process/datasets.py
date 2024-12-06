from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from skimage.transform import resize
import torch
from utils import xt_load_rawdata, StandardScaler, adj_to_laplacian

class EEGDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, freq=256, index_col=None, use_freq=False, transform=None):
        """
        Arguments:
            data (pandas.DataFrame): list if data directory and labels
            freq (int, optional): frequency of data points. Defaults to 256
            index_col (int, optional): if input data contain index. Defaults to None
            use_freq (bool, optional): generate frequent-domain features or not
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.freq = freq
        self.index_col = index_col
        self.use_freq = use_freq
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rows = self.data.iloc[idx]
        data_dir = '../' + rows['data_dir']
        # adj_dir = '../' + rows['adj_dir']
        data = pd.read_csv(data_dir, header=None, index_col=self.index_col).values
        data = resize(data, (data.shape[0], self.freq * 10))
        if self.use_freq:
            data = xt_load_rawdata(data, filter_en=1, resample_en=0)
        # adj = pd.read_csv(adj_dir, header=0, index_col=0).values
        # assert adj.shape == (data.shape[0], data.shape[0]), f'{adj_dir} adj shape {adj.shape}'
        if self.transform:
            data = self.transform.transform(data)

        data = torch.tensor(data).unsqueeze(dim=0).float()
        adj = torch.zeros(1) #torch.tensor(adj).float()
        label = torch.tensor(rows['label']).long()

        return data, adj, label

class EEGDataset_preload(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, freq=256, index_col=None, use_freq=False, transform=None):
        """
        load all data during initialization
        Arguments:
            data (pandas.DataFrame): list if data directory and labels
            freq (int, optional): frequency of data points. Defaults to 256
            index_col (int, optional): if input data contain index. Defaults to None
            use_freq (bool, optional): generate frequent-domain features or not
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.freq = freq
        self.index_col = index_col
        self.use_freq = use_freq
        self.x, self.adj, self.y, self.ids = [], [], [], []
        self.scalers = {}
        for i, row in self.data.iterrows():
            # data_dir = row['data_dir'][3:] #'../' + row['data_dir']
            # # adj_dir = '../' + row['adj_dir']
            # cur = pd.read_csv(data_dir, header=None, index_col=index_col).values
            data_dir = row['data_dir'][3:-4] + '_pad.npy'  # '../' + row['data_dir']
            # adj_dir = '../' + row['adj_dir']
            cur = np.load(data_dir)
            try:
                cur = resize(cur, (cur.shape[0], freq * 10))
                if np.all(np.isclose(cur, cur[0, 0])):
                    raise ValueError('values are too small')
                if self.use_freq:
                    cur = xt_load_rawdata(cur, filter_en=1, resample_en=0)
                scaler = StandardScaler(mean=cur.mean(), std=cur.std())
                cur = scaler.transform(cur)
                adj = np.corrcoef(cur)
                adj = adj_to_laplacian(adj)
            except:
                print('fail loading data in ' + data_dir + ', skipping the file...')
                continue
            # adj = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((adj.row, adj.col))), torch.FloatTensor(adj.data), torch.Size(adj.shape))

            self.x.append(cur)
            self.adj.append(adj)
            self.y.append(row['label'])
            self.ids.append(i)
            self.scalers[i] = scaler

        self.x = np.stack(self.x, axis=0)
        self.adj = np.stack(self.adj, axis=0)
        self.y, self.ids = np.array(self.y), np.array(self.ids)
        self.mean, self.std = self.x.mean(), self.x.std()
        self.transform = transform if transform is not None else StandardScaler(mean=self.mean, std=self.std)
        self.x = self.transform.transform(self.x)
        print('dataset metadata:', self.mean, self.std, f"{self.y.sum()}/{self.y.shape}", self.x.shape, self.adj.shape)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ids = torch.tensor(self.ids[idx]).long()
        data = torch.tensor(self.x[idx]).unsqueeze(dim=0).float()
        adj = torch.tensor(self.adj[idx]).float()
        label = torch.tensor(self.y[idx]).long()

        return data, adj, label, ids

class EEGDataset_preload_ode(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, freq=256, index_col=None, use_freq=False, transform=None):
        """
        load all data during initialization
        Arguments:
            data (pandas.DataFrame): list if data directory and labels
            freq (int, optional): frequency of data points. Defaults to 256
            index_col (int, optional): if input data contain index. Defaults to None
            use_freq (bool, optional): generate frequent-domain features or not
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.freq = freq
        self.index_col = index_col
        self.use_freq = use_freq
        self.x, self.y, self.ids = [], [], []
        self.scalers = []
        for i, row in self.data.iterrows():
            data_dir = row['data_dir'][3:] # '../' + row['data_dir']#'../' + row['data_dir']#row['data_dir'][6:] # '../' + row['data_dir']
            cur = pd.read_csv(data_dir, header=None, index_col=index_col).values
            cur = resize(cur, (cur.shape[0], freq * 10))
            if np.all(np.isclose(cur, cur[0, 0])):
                continue
            if self.use_freq:
                cur = xt_load_rawdata(cur, filter_en=1, resample_en=0)
            scaler = StandardScaler(mean=cur.mean(), std=cur.std())
            cur = scaler.transform(cur)
            self.x.append(cur)
            self.scalers.append(scaler)
            self.y.append(row['label'])
            self.ids.append(i)
        self.x = np.stack(self.x, axis=0)
        self.y, self.ids = np.array(self.y), np.array(self.ids)
        self.mean, self.std = self.x.mean(), self.x.std()
        self.transform = transform if transform is not None else StandardScaler(mean=self.mean, std=self.std)
        # self.x = self.transform.transform(self.x)
        print('dataset metadata:', self.mean, self.std, f"{self.y.sum()}/{self.y.shape}", self.x.shape)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ids = torch.tensor(self.ids[idx]).long()
        data = torch.tensor(self.x[idx]).float()
        label = torch.tensor(self.y[idx]).long()

        return data, label, ids

class EEGDataset_raw(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, freq=256, index_col=None, use_freq=False, transform=None):
        """
        assume data contains the patient ids, signals, and labels
        Arguments:
            data (tuple[numpy.array, numpy.array, numpy.array]): ids, signals, and labels
            freq (int, optional): frequency of data points. Defaults to 256
            index_col (int, optional): dummy. Defaults to None
            use_freq (bool, optional): generate frequent-domain features or not
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.freq = freq
        self.index_col = index_col
        self.use_freq = use_freq
        self.x, self.adj, self.y, self.sids = data[1], [], data[2], data[0]
        for i, cur in enumerate(self.x):
            try:
                if np.all(np.isclose(cur, cur[0, 0])):
                    raise ValueError('values are too small')
                if self.use_freq:
                    cur = xt_load_rawdata(cur, filter_en=1, resample_en=0)
                adj = np.corrcoef(cur)
                adj = adj_to_laplacian(adj)
            except:
                print('fail generating adj of instance ' + i + ', discard instance...')
                continue
            # adj = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((adj.row, adj.col))), torch.FloatTensor(adj.data), torch.Size(adj.shape))

            self.adj.append(adj)
        self.adj = np.stack(self.adj, axis=0)
        self.mean, self.std = self.x.mean(), self.x.std()
        self.transform = transform if transform is not None else StandardScaler(mean=self.mean, std=self.std)
        self.x = self.transform.transform(self.x)
        print('dataset metadata:', self.mean, self.std, f"{self.y.sum()}/{self.y.shape}", self.x.shape, self.adj.shape)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.tensor(self.x[idx]).unsqueeze(dim=0).float()
        adj = torch.tensor(self.adj[idx]).float()
        label = torch.tensor(self.y[idx]).long()

        return data, adj, label

