import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_process.load_dataset import load_mat_ode
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils import reconstruction_errors, save_data, plot_signal, inverse_unfold
from configs import split_ids
import random
import copy
# from data_process.ODE_utils import *
from networks.latentODE import latentODEVAE

# check CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('torch version: ', torch.__version__)
print('GPU State:', device)

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=100000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--train_dir', type=str, default='models')
# data chbmit
parser.add_argument('-dataset', type=str, default='chbmit_mini', help='choose dataset to run [options: TUSZ_mini, chbmit_mini]')

args = parser.parse_args()


if __name__ == '__main__':
    suffix = '_pad'
    ckpt_path = os.path.join(args.train_dir, f'ckpt_latentODE_{args.dataset}{suffix}.pth')
    training = True
    epochs = args.niters
    batch_size = 256
    lr = args.lr
    step_size = 50
    early_stop = step_size
    noise_std = 0.05

    n_channel = 23 if 'chbmit' in args.dataset else 20

    dataloader, transform, meta = load_mat_ode(args.dataset, os.path.join('../../data/EEG/', args.dataset), BATCH_SIZE=batch_size)  # , use_freq=True)#, transform=scaler)#, return_data=True)

    # 1 as feature dimension
    obs_dim = 1
    latent_dim = 32

    nhidden = 64
    start = 0.
    stop = 1.
    l_window, l_overlap, l_pad = 250, 50, 150

    # model
    net = latentODEVAE(obs_dim, nhidden, latent_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=step_size, eta_min=0.)#StepLR(optimizer, step_size=step_size, gamma=gamma)
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, weights_only=True)
            net.load_state_dict(checkpoint['net_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    if training:
        net.train()
        best_epoch, best_loss = -1, np.inf
        for itr in range(1, epochs + 1):
            train_loss, cur_loss = 0, 0
            for x, y, ids in dataloader:
                optimizer.zero_grad()
                x = x.permute(0, 2, 1, 3)[:, :, [0]].to(device) # B, N, T -> B, T, N
                x = x.repeat(batch_size, 1, 1)
                x = x + torch.normal(0, noise_std, size=x.shape).to(device)

                B, T, N = x.shape
                x = F.pad(x, (0, 0, 0, l_pad), mode='reflect') # B, T+l_pad, N
                x = x.unfold(1, l_window, l_window-l_overlap) # B, (T+l_pad-l_window)/(l_window-l_overlap), N, l_window
                T_p = (T + l_pad - l_window) // (l_window - l_overlap) + 1
                x = x.reshape(B * T_p, N, l_window).permute(2, 0, 1)  # l_window, B*T_p, N

                T, B, N = x.shape

                # 1 as feature dimension
                t = torch.linspace(start, stop, T).to(device)#.unsqueeze(-1).repeat(1, B*N).unsqueeze(-1) # T, B*N, 1
                max_len = np.random.choice([T // 8, T // 4, T // 2, T])
                permutation = torch.randperm(T)
                permutation, _ = torch.sort(permutation[:max_len])

                x, t = x[permutation], t[permutation]
                x_p, z, z_mean, z_log_var = net(x, t)

                # 1 as feature dimension
                kl_loss = -0.5 * torch.sum((1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)).reshape(B, N*latent_dim), -1)

                rec_loss = 0.5 * ((x - x_p) ** 2).reshape(T, B, N).sum(-1).sum(0) / noise_std**2
                loss = rec_loss + kl_loss
                loss = torch.mean(loss)
                loss /= T
                loss.backward()  # backward
                optimizer.step()
                train_loss += loss.data
                cur_loss += rec_loss.mean().data / T

            if itr % 10 == 0:
                # plot_signal(x[:, 0, 0].detach().cpu().numpy(), x_p[:, 0, 0].detach().cpu().numpy(), t.detach().cpu().numpy(), itr//10)
                plot_signal(x.squeeze().mean(-1).detach().cpu().numpy(), x_p.squeeze().mean(-1).detach().cpu().numpy(),
                            t.detach().cpu().numpy(), itr // 10)
            print('Iter: {}, shape: {}, running avg elbo: {:.4f}, reconstruct: {:.4f}'
                  .format(itr, x.shape, train_loss/len(dataloader), cur_loss/len(dataloader)))
            scheduler.step()
            if best_loss > cur_loss:
                best_epoch, best_loss = itr, cur_loss
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))

            if itr - best_epoch > early_stop:
                print(f'Early stop at iteration: {itr}')
                break
        print('Training complete after {} iters.'.format(itr))

    ## generate denoised data
    mses, mapes, rmses = [], [], []

    net.eval()
    for x, y, ids in dataloader:
        x = x.permute(0, 2, 1, 3)[:, :, [0]].to(device)  # B, N, T -> B, T, N
        x = x.repeat(batch_size, 1, 1)

        B0, T0, N0 = x.shape
        x = F.pad(x, (0, 0, 0, l_pad), mode='reflect')  # B, T+l_pad, N
        x = x.unfold(1, l_window, l_window - l_overlap)  # B, (T+l_pad-l_window)/(l_window-l_overlap), N, l_window
        T_p = (T0 + l_pad - l_window) // (l_window - l_overlap) + 1
        x = x.reshape(B0 * T_p, N0, l_window).permute(2, 0, 1)  # l_window, B*T_p, N

        T, B, N = x.shape

        # channel as feature dimension
        t = torch.linspace(start, stop, T).to(device) # T

        x_p, z, z_mean, z_log_var = net(x, t, MAP=True)
        # res = x_p.permute(1, 2, 0).detach().cpu().numpy()
        res = x_p.reshape(l_window, B0, T_p, N).permute(1, 2, 0, 3).detach().cpu().numpy() # B, T_p, l_window, N
        res = inverse_unfold(res, l_pad, l_overlap)
        print('result shape', res.shape)

        mse, mape, mae, r2 = reconstruction_errors(x.flatten().detach().cpu().numpy(), x_p.flatten().detach().cpu().numpy())
        save_data(res, ids, dataloader, suffix)
        print(f"MSE: {mse}, MAPE: {mape}, MAE: {mae}, R2: {r2}")


