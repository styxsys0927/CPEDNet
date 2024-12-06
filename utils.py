import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sklearn
import sklearn.metrics
from scipy.sparse.linalg import eigsh
import sys
from scipy.signal import butter, lfilter, freqz, sosfilt, filtfilt
import scipy.io as spio
import matplotlib.pyplot as plt


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return sparse_to_tuple(features)

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1, keepdims=True)
    B_mB = B - B.mean(-1, keepdims=True)

    # Sum of squares across rows
    ssA = (A_mA**2).sum(-1, keepdims=True)
    ssB = (B_mB**2).sum(-1, keepdims=True)
    print(A_mA.shape, B_mB.shape, ssA.shape, ssB.shape)
    cov = np.einsum('bnt,btm->bnm', A_mA, B_mB.transpose(0, -1, -2))
    ssp = np.einsum('bnt,btm->bnm', ssA, ssB.transpose(0, -1, -2))
    # Finally get corr coeff
    return cov / np.sqrt(ssp)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(np.abs(adj))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def adj_to_laplacian(adj):
    adj = adj - sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    return np.array(scaled_laplacian.todense())


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def corr2_coeff(A, B):
    # 3d tensor corr by the last two
    A_mA = A - A.mean(-2)[:, None]
    B_mB = B - B.mean(-2)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(-2)
    ssB = (B_mB**2).sum(-2)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = butter(order, [low, high], btype='band', output='sos')
    # return b, a
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

# Low-pass filter with 120 Hz cutoff frequency
def lowpass_filter(data, cutoff_freq, fs, order=5):
    # Design a Butterworth high-pass filter
    b, a = butter(order, cutoff_freq, btype='lowpass', fs=fs, output='ba')
    # filtered_data = filtfilt(b, a, data)  # Apply filter
    filtered_data = np.concatenate([filtfilt(b, a, data[i, :, 0]) for i in range(data.shape[0])]).reshape(data.shape[0], data.shape[1],
                                                                                          1)
    return filtered_data

def xt_load_rawdata(x, filter_en, resample_en):
    if filter_en == 1:
        n_band = 5
    else:
        n_band = 1

    n_channel = x.shape[0]
    sfreq = x.shape[1]
    # Resample patient 5000 Hz to 500 Hz
    if resample_en == 1:
        X = x[:, ::10]
    else:
        s_freq = sfreq
        if filter_en == 1:
            # Band-pass filter
            order = 6
            lowcut_freq = [1, 4, 8, 12, 30]
            highcut_freq = [4, 8, 12, 30, 64]
            X = np.zeros((n_channel * n_band, s_freq))
            for j in range(0, n_channel):
                kk = j * 5
                for k in range(0, 5):
                    X[kk + k] = butter_bandpass_filter(x[j], lowcut_freq[k], highcut_freq[k], s_freq, order)
        else:
            X = x

    return X

def show_results(y_truth, y_pred):

    # print(np.sum(y_truth), np.sum(y_pred))
    accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
    f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=0))
    prec_score = (sklearn.metrics.precision_score(y_truth, y_pred, pos_label=0))
    recall_score = (sklearn.metrics.recall_score(y_truth, y_pred, pos_label=0))
    auc_score = (sklearn.metrics.roc_auc_score(y_truth, y_pred))
    # f1_ma = (sklearn.metrics.f1_score(y_truth, y_pred, average='macro'))
    # f1_mi = (sklearn.metrics.f1_score(y_truth, y_pred, average='micro'))
    # prec_score = (sklearn.metrics.precision_score(y_truth, y_pred, average='micro'))
    # recall_score = (sklearn.metrics.recall_score(y_truth, y_pred, average='micro'))
    # confusion_mat = (sklearn.metrics.confusion_matrix(y_truth, y_pred))

    return accuracy_score, f1_score, prec_score, recall_score, auc_score

def reconstruction_errors(y_true, y_pred):
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    return mse, mape, mae, r2

def save_data(data, ids, dataloader, suffix, transform=True):
    ids = ids.tolist()
    print(ids)
    dirs = dataloader.dataset.data.loc[ids]['data_dir']
    scalers = dataloader.dataset.scalers
    for i, dir in enumerate(dirs):
        dir = dir[3:-4] + suffix #'_latentODE_1_single.csv'
        # print(dir)
        cur = data[i] if not transform else scalers[ids[i]].inverse_transform(data[i])
        np.save(dir, cur, allow_pickle=True)

def plot_signal(y_true, y_pred, t, itr):
    fig = plt.figure(figsize=(8, 3), num=1, clear=True)
    ax = fig.add_subplot()

    ax.plot(t, y_true, label='Original EEG')
    ax.plot(t, y_pred, label='Reconstructed EEG')
    plt.legend()
    plt.savefig(f'./results/{itr}_pad.png')
    # plt.show()

def inverse_unfold(x, l_pad, l_overlap):
    assert l_overlap > 1, "l_overlap must be larger than 1"
    res = x[:, 0] # B, l_window, N
    kernel = np.arange(0, 1+1/l_overlap, 1/(l_overlap-1))
    for i in range(1, x.shape[1]):
        cur = x[:, i]
        cur[:, :l_overlap] = cur[:, :l_overlap] * kernel[None, :, None]
        res[:, -l_overlap:] = res[:, -l_overlap:] * kernel[None, ::-1, None] + cur[:, :l_overlap]
        res = np.concatenate([res, cur[:, l_overlap:]], axis=1)
    res = res[:, :-l_pad]
    return res
