import numpy as np
import matplotlib.pyplot as plt
import random
from data_process.load_dataset import load_mat
from utils import show_results, StandardScaler, save_data
import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
import argparse
import os
import copy
from configs import split_ids
from networks.transformer import TranAD
from networks.gcn import ChebNetGCN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

import warnings 
warnings.filterwarnings("ignore")

# check CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('torch version: ', torch.__version__)
print('GPU State:', device)

# Settings
parser = argparse.ArgumentParser(description='traffic prediction')

# data chbmit
parser.add_argument('-dataset', type=str, default='TUSZ_mini', help='choose dataset to run [options: TUSZ_mini, chbmit_mini]')

# model
parser.add_argument('-model', type=str, default='CPEDNet', help='choose model to train and test [options: LightK_DSGCN, DSGCN, ncDSGCN, ntDSGCN, nsDSGCN]')
parser.add_argument('-train', type=str, default='False', help='True if the model needs to be trained')
parser.add_argument('-model_path', type=str, default='./models/TUSZ_mini_EEGNet_ELU', help='choose model parameters to load')
parser.add_argument('-save_path', type=str, default='./models/TUSZ_mini_EEGNet_ELU', help='location to save the new model')
parser.add_argument('-learning_rate', type=float, default=0.00005, help='Initial learning rate.')  # 0.01
parser.add_argument('-epochs', type=int, default=500, help='Number of epochs to train.')  # 300
parser.add_argument('-hidden1', type=int, default=32, help='Number of units in hidden layer 1.')  # 16
parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')  # 0.5
parser.add_argument('-weight_decay', type=float, default=1e-4, help='Weight for L2 loss on embedding matrix.')  # 5e-4
parser.add_argument('-early_stopping', type=int, default=20, help='Tolerance for early stopping (# of steps).')
parser.add_argument('-max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')  # 3
args = parser.parse_args()

suffix = '_CPEDNet_filtered'#'_CPEDNet_'#

epochs = args.epochs
batch_size = 2
lr = args.learning_rate
step_size = 30
gamma = 0.9 # gamma of scheduler
early_stop = args.early_stopping
n_hidden = args.hidden1
l_window = 10

# chbmit
# n_channel = 23
# n_time = 256

# TUSZ
n_channel = 20
n_time = 250

# CPEDNet
class CPEDNet(nn.Module):
    def __init__(self, n_channel, n_time, n_in, n_hidden, l_window, dropout=0.2):
        super(CPEDNet, self).__init__()
        self.l_window = l_window
        self.in_conv = nn.Sequential(
            nn.Conv2d(n_in, n_hidden, kernel_size=(1, 1)),
            nn.BatchNorm2d(n_hidden), # B, C, N, T
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        self.gcn = ChebNetGCN(n_hidden, n_hidden, n_hidden, dropout=0.2, k=4)

        self.transformer = TranAD(n_hidden, l_window, n_forward=16, dropout=dropout)

        self.out_ffn = nn.Sequential(
            nn.Linear(n_hidden*3, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 2)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(n_hidden*3, n_hidden*3, kernel_size=(1, 50), stride=(1, 50)),
            nn.BatchNorm2d(n_hidden*3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(n_channel, (n_time*10-50)//50+1)),
            nn.Dropout(p=0.2)
        )

    def forward(self, x, adj):
        # assert torch.isnan(x).sum() == 0 or torch.isnan(adj).sum() == 0, f"input {torch.isnan(x).sum()}, {torch.isnan(adj).sum()}, {adj}"
        x = self.in_conv(x) # B, C, N, T
        # assert torch.isnan(x).sum() == 0, f"in_conv {torch.isnan(x).sum()}"
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)
        x = self.gcn(x, adj)
        # assert torch.isnan(x).sum() == 0, f"gcn {torch.isnan(x).sum()}"
        x = x.reshape(B, T, N, C).transpose(1, 2).reshape(B*N, T, C)
        x0 = F.pad(x, (0, 0, self.l_window-1, 0), mode='reflect')
        x0 = x0.unfold(1, self.l_window, 1)
        x0 = x0.reshape(B*N*T, C, self.l_window).permute(2, 0, 1) # K, B*N*T, C
        # assert torch.isnan(x0).sum() == 0, f"x0 {torch.isnan(x0).sum()}"
        x1, x2 = self.transformer(x0, x0[[-1]])
        x1, x2 = x1.reshape(B, N, T, C), x2.reshape(B, N, T, C)
        # assert torch.isnan(x1).sum() == 0, f"x1 {torch.isnan(x1).sum()}"
        # assert torch.isnan(x2).sum() == 0, f"x2 {torch.isnan(x2).sum()}"
        out = torch.concat([x.reshape(B, N, T, C), x1, x2], dim=-1).permute(0, 3, 1, 2)
        out = self.out_conv(out).squeeze((-2, -1)) # B, N,
        out = self.out_ffn(out)

        # assert torch.isnan(out).sum() == 0, f"out {torch.isnan(out).sum()}"
        return x1, x2, out

# Evaluatoin Function
def evaluate(model, data, save_name=None):
    model.eval()
    preds, labels = [], []
    for x, adj, y, ids in data:
        x_batch, y_batch = x.to(device), y.to(device)
        adj = adj.to(device)
        # print('Eval:', x_batch.mean().item(), x_batch.std().item(), y_batch.sum().item())
        x1, x2, pred = net(x_batch, adj)  # forward
        score = 0.5*((x1**2).mean(-1)+(x2**2).mean(-1))
        if save_name is not None:
            save_data(score.detach().cpu().numpy(), ids, data, save_name+'_score', transform=False)
        pred = np.array(pred.data.cpu().numpy()).argmax(axis=1)
        preds.append(pred)
        labels.append(y)
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    results = show_results(labels, preds)
    if save_name is not None:
        np.save(save_name+'_pred', preds, allow_pickle=True)
        np.save(save_name+'_labels', preds, allow_pickle=True)
    print('Tested: Acc: {}, F1: {}, PRC: {}, RCL: {}, AUC: {}'.format(results[0], results[1], results[2], results[3],
                                                                      results[4]))
    return results[1] #accuracy_score(y, np.array(pred.data.cpu().numpy()).argmax(axis=1))

# Model Training
def model_train(net, train, test, epochs, batch_size, scheduler, suffix):
    # train_acc = []
    # test_acc = []
    best_epoch, best_f1 = 0, 0
    for epoch in range(1, epochs+1):
        net.train(mode=True)
        train_loss = 0.0
        bid = 0
        for x, adj, y, ids in train:
            x_batch, y_batch = x.to(device), y.to(device)
            adj = adj.to(device)
            # print('Train:', x_batch.mean().item(), x_batch.std().item(), y_batch.sum().item())
            optimizer.zero_grad() # zero the parameter gradients
            x1, x2, outputs = net(x_batch, adj) # forward
            # loss = (1/epoch*((x1.permute(0, 3, 1, 2)-x_batch)**2).mean() + (1-1/epoch)*((x2.permute(0, 3, 1, 2)-x_batch)**2).mean())

            loss = 1/epoch*(x1**2).mean() + (1-1/epoch)*(x2**2).mean()
            loss += criterion(outputs, y_batch) # calculate loss
            # print('loss all', loss.data.cpu().numpy())
            loss.backward() # backward
            optimizer.step()
            train_loss += loss.data
            bid += 1

        # train_acc.append(evaluate(net, train))
        f1 = evaluate(net, test)
        print(f'Epoch: {epoch}/{epochs} -- Loss: {train_loss/(bid+1)}; F1: {f1}, Best epoch: {best_epoch}')
        scheduler.step()            
        if best_f1 < f1:
            best_f1 = f1
            best_epoch = epoch
            if not os.path.exists('./models'):
                os.mkdir('./models')
            torch.save(net.state_dict(), f'./models/{args.dataset}{suffix}.pth')

        if epoch - best_epoch > early_stop:
            print('Early stopping at {}'.format(best_epoch))
            break
        # accuracy result every 50 epoch
        # if epoch % 50 == 0:
        #     print("\nEpoch ", epoch)
        #     print("(Training Loss) -", train_loss.data.cpu().numpy())
        #     print("(Train) - ", evaluate(net, X_train, y_train))
        #     print("(Test) - ", evaluate(net, X_test, y_test))
            
    return best_f1 #train_acc, test_acc

# activation_box = ['ReLU', 'LeakyReLU', 'ELU', 'PReLU'] # activation function you want to try
activation_box = ['ELU'] # activation function you want to try

train_acc, test_acc = {}, {}
best_train_acc, best_test_acc = {}, {}

net = CPEDNet(n_channel=n_channel, n_time=n_time, n_in=1, n_hidden=n_hidden, l_window=l_window).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
print('#####################################################')
input_size = [(1, n_channel, n_time*10), (n_channel, n_channel)]
try:
    summary(net, input_size=input_size)
except:
    pass
print('#####################################################')

train_loader, val_loader, test_loader, transform = load_mat(args.dataset, os.path.join('../../data/EEG/', args.dataset),
                                                 split_ids=split_ids, BATCH_SIZE=batch_size)
if args.train == 'True':
    best_f1 = model_train(net, train_loader, val_loader, epochs=epochs,
                                      batch_size=batch_size, scheduler=scheduler, suffix=suffix)

checkpoint = torch.load(f'./models/{args.dataset}{suffix}.pth', weights_only=True)
net.load_state_dict(checkpoint)
evaluate(net, test_loader, f'test_{args.dataset}{suffix}')