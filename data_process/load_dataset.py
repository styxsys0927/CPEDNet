import sys

sys.path.append('..')
import scipy.io as sio
import os, glob, random, gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from utils import *
from data_process.datasets import EEGDataset, EEGDataset_preload, DataLoader, EEGDataset_preload_ode, EEGDataset_raw
from skimage.transform import resize


def load_mat(dataset, data_dir, split_ids=None, BATCH_SIZE=32, transform=None, use_freq=False, return_data=False):
    data_all = {'train': [], 'eval': [], 'dev': []}
    if dataset == 'chbmit_mini':
        assert split_ids is not None, 'split_ids must be specified for chbmit_mini'
        if os.path.exists(os.path.join(data_dir, 'data.csv')):
            data = pd.read_csv(os.path.join(data_dir, 'data.csv'), header=0)
        else:
            data = []
            sid_dict = set()
            for file in glob.iglob(data_dir + '/**', recursive=True):
                if file.endswith(".csv") and (not file.endswith("_matrix.csv")):
                    file = file.replace('\\', '/')
                    subject_id = file.split('/')[-2].split('_')[0]
                    if '00_epilepsy' in file:
                        data.append(
                            {'data_dir': file, 'adj_dir': file[:-4] + "_matrix.csv", 'subject_id': subject_id,
                             'label': 0})
                    else:
                        data.append(
                            {'data_dir': file, 'adj_dir': file[:-4] + "_matrix.csv", 'subject_id': subject_id,
                             'label': 1})
                    sid_dict.add(subject_id)
            data = pd.DataFrame(data)
            sid_dict = sorted(list(sid_dict))
            sid_dict = {sid_dict[i]: i for i in range(len(sid_dict))}
            data['sid'] = data['subject_id'].apply(lambda x: sid_dict[x])
            data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)

        data = data.sample(frac=1)
        n_train, n_dev = int(data.shape[0] * 0.8), int(data.shape[0] * 0.1)
        data_all['train'] = data.iloc[:n_train]
        data_all['dev'] = data.iloc[n_train:(n_train + n_dev)]
        data_all['eval'] = data.iloc[n_train + n_dev:]
        # data_all['train'] = data[data['sid'].isin(split_ids[0])]
        # data_all['dev'] = data[data['sid'].isin(split_ids[1])]
        # data_all['eval'] = data[data['sid'].isin(split_ids[2])]
    elif dataset == 'TUSZ_mini':
        assert split_ids is not None, 'split_ids must be specified for TUSZ_mini'
        if os.path.exists(os.path.join(data_dir, 'data.csv')):
            data = pd.read_csv(os.path.join(data_dir, 'data.csv'), header=0)
        else:
            data = []
            sid_dict = set()
            for file in glob.iglob(data_dir + '/**', recursive=True):
                if file.endswith(".csv") and (not file.endswith("_matrix.csv")):
                    file = file.replace('\\', '/')
                    subject_id = file.split('/')[-2]
                    if '00_epilepsy' in file:
                        data.append(
                            {'data_dir': file, 'adj_dir': file[:-4] + "_matrix.csv", 'subject_id': subject_id,
                             'label': 0})
                    else:
                        data.append(
                            {'data_dir': file, 'adj_dir': file[:-4] + "_matrix.csv", 'subject_id': subject_id,
                             'label': 1})
                    sid_dict.add(subject_id)
            data = pd.DataFrame(data)
            sid_dict = sorted(list(sid_dict))
            sid_dict = {sid_dict[i]: i for i in range(len(sid_dict))}
            # print(data)
            data['sid'] = data['subject_id'].apply(lambda x: sid_dict[x])
            data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)

        data = data.sample(frac=1)
        n_train, n_dev = int(data.shape[0] * 0.8), int(data.shape[0] * 0.1)
        data_all['train'] = data.iloc[:n_train]
        data_all['dev'] = data.iloc[n_train:(n_train + n_dev)]
        data_all['eval'] = data.iloc[n_train + n_dev:]
        # data_all['train'] = data[data['sid'].isin(split_ids[0])]
        # data_all['dev'] = data[data['sid'].isin(split_ids[1])]
        # data_all['eval'] = data[data['sid'].isin(split_ids[2])]
    elif dataset == 'TUSZ':
        sid_dict = set()
        for set_name in ['train', 'dev', 'eval']:
            if os.path.exists(os.path.join(data_dir, f'{set_name}_data.csv')):
                data = pd.read_csv(os.path.join(data_dir, f'{set_name}_data.csv'), index_col=0)
            else:
                data = []
                for file in glob.iglob(data_dir + f'{set_name}/**', recursive=True):
                    if file.endswith(".csv") and (not file.endswith("_matrix.csv")):
                        if '00_epilepsy' in file:
                            data.append(
                                {'data_dir': file, 'adj_dir': file[:-4] + "_matrix.csv",
                                 'subject_id': file.split('/')[-2],
                                 'label': 0})
                        else:
                            data.append(
                                {'data_dir': file, 'adj_dir': file[:-4] + "_matrix.csv",
                                 'subject_id': file.split('/')[-2],
                                 'label': 1})
                        sid_dict.add(file.split('/')[-2])
                data = pd.DataFrame(data)
            data_all[set_name] = data

        sid_dict = sorted(list(sid_dict))
        sid_dict = {sid_dict[i]: i for i in range(len(sid_dict))}
        for set_name in ['train', 'dev', 'eval']:
            if not os.path.exists(os.path.join(data_dir, f'{set_name}_data.csv')):
                data_all[set_name]['sid'] = data_all[set_name]['subject_id'].apply(lambda x: sid_dict[x])
                data_all[set_name].to_csv(os.path.join(data_dir, f'{set_name}_data.csv'), index=False)
    else:
        print('Dataset is not allowed')
        quit()

    print(
        f"{dataset} -- total number of samples {data.shape[0]} with {data['label'].sum()} normal samples of {data['sid'].unique().shape[0]} subjects")

    if 'chbmit' in dataset:
        freq = 256
        index_col = None
    elif 'TUSZ' in dataset:
        freq = 250
        index_col = 0
    else:
        freq = 500
        index_col = None

    if return_data:  # used for ML
        datasets = []
        for set_name in ['train', 'dev', 'eval']:
            x = []
            for i, row in data_all[set_name].iterrows():
                cur = pd.read_csv('../' + row['data_dir'], header=None, index_col=index_col).values
                cur = resize(cur, (cur.shape[0], freq * 10))
                # cur = np.reshape(cur, (-1, freq))
                # cur = cur.sum(axis=0)
                # assert cur.shape == (23, 2560), f"{row['data_dir']} has shape {cur.shape}"
                x.append(cur)
            x = np.stack(x, axis=0)
            print(set_name, x.mean(), x.std())
            datasets.append((x, data_all[set_name]['label'].values))
        return datasets

        # scaler = StandardScaler(X_mean, X_std)
    train_dataset = EEGDataset_preload(data_all['train'], freq=freq, index_col=index_col, use_freq=use_freq,
                                       transform=transform)
    # if transform is None:
    #     transform = StandardScaler(train_dataset.mean, train_dataset.std)
    valid_dataset = EEGDataset_preload(data_all['eval'], freq=freq, index_col=index_col, use_freq=use_freq,
                                       transform=transform)
    test_dataset = EEGDataset_preload(data_all['dev'], freq=freq, index_col=index_col, use_freq=use_freq,
                                      transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, transform


def load_mat_ode(name, datadir, BATCH_SIZE=32, transform=None, use_freq=False, return_data=False):
    freqs, index_cols = {'chbmit_mini': 256, 'TUSZ_mini': 250}, {'chbmit_mini': None, 'TUSZ_mini': 0}
    data = pd.read_csv(os.path.join(datadir, 'data.csv'), header=0)
    dataset = EEGDataset_preload_ode(data, freq=freqs[name], index_col=index_cols[name], use_freq=use_freq,
                                     transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    transform = dataset.transform
    return dataloader, transform, data

    # data_dir = '../../../data/EEG/chbmit_mini'
# load_mat('chbmit_mini', data_dir, split_ids=[list(range(19)), [19, 20], [21, 22, 23]])
# data_dir = '../../../data/EEG/TUSZ_mini'
# load_mat('TUSZ_mini', data_dir, split_ids=[list(range(42)), [42, 43, 44, 45, 46], [47, 48, 49, 50, 51, 52]])
