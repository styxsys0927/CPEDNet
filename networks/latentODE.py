import math
import numpy as np
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable

from torchdiffeq import odeint_adjoint as odeint


use_cuda = torch.cuda.is_available()

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = odeint(self.func, z0, t, method='dopri8')
        # z = odeint(self.func, z0, t, method='scipy_solver', options={"solver": 'LSODA'})
        if return_whole_sequence:
            return z
        else:
            return z[-1]

class NNODEF(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers=5, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant
        self.nfe = 0

        if time_invariant:
            self.lin_in = nn.Linear(in_dim, hid_dim)
        else:
            self.lin_in = nn.Linear(in_dim+1, hid_dim)
        self.lin_out = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)
        self.mlp = nn.ModuleList()
        for i in range(n_layers):
            self.mlp.append(
                nn.Sequential(
                    nn.LayerNorm(hid_dim),
                    nn.Linear(hid_dim, hid_dim),
                    nn.ELU(inplace=True)
                )
            )
        self.norm = nn.LayerNorm(hid_dim)
        y = 1.0 / np.sqrt(n_layers+1)
        self.weights = nn.Parameter(torch.ones(n_layers+1)).data.uniform_(-y, y).cuda()

    def forward(self, t, x):
        self.nfe += 1
        if not self.time_invariant:
            x = torch.cat((x, t.repeat(x.shape[0]).unsqueeze(-1)), dim=-1)
            # x = torch.cat((x, t), dim=-1)

        hs = [self.elu(self.lin_in(x))]
        for mlp in self.mlp:
            h = mlp(hs[-1])
            hs.append(h)

        hs = (torch.stack(hs, 0) * self.weights[:, None, None]).sum(0)
        hs = self.norm(hs)
        out = self.lin_out(hs)
        return out

# class GRU(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_layers=3):
#         super(GRU, self).__init__()
#         self.in_gru = nn.GRUCell(input_dim+1, hidden_dim)
#         self.grus = nn.ModuleList()
#         for i in range(n_layers):
#             self.grus.append(
#                 nn.Sequential(
#                     nn.LayerNorm(hidden_dim),
#                     nn.GRUCell(hidden_dim, hidden_dim),
#                 )
#             )
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.weights = nn.Parameter(torch.tensor([1] + [0] * n_layers))
#     def forward(self, x, h):
#         hs = [self.in_gru(x, h)]
#         for mlp in self.mlp:
#             h = mlp(hs[-1])
#             hs.append(h)
#
#         hs = (torch.cat(hs, 0) * self.weights[:, None, None, None]).sum(0)
#         hs = self.norm(hs)
#         out = self.lin_out(hs)
#
#         return out, h

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        num_layers = 3
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.rnn = nn.GRU(input_dim+1, hidden_dim, num_layers=num_layers)#, dropout=0.2)
        self.norm = nn.LayerNorm(hidden_dim)# nn.RMSNorm(hidden_dim)
        y = 1.0 / np.sqrt(num_layers+1)
        self.weights = nn.Parameter(torch.ones(num_layers+1)).data.uniform_(-y, y).cuda()
        self.hid2lat = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2*latent_dim)
        )

    def forward(self, x, t):
        # Concatenate time to input
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        # xt = torch.cat((x, t), dim=-1)
        xt = torch.cat((x, t[:, None, None].repeat(1, x.shape[1], 1)), dim=-1)
        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # print('h0',h0.shape)
        hs = (torch.cat((x[[0]], h0), 0) * self.weights[:, None, None]).sum(0)
        h0 = self.norm(hs)
        # Compute latent dimension
        z0 = self.hid2lat(h0)
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        return z0_mean, z0_log_var

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.norm = nn.LayerNorm(latent_dim)#nn.RMSNorm(latent_dim)
        self.l2o = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z0, t):
        zs = self.ode(z0, t, return_whole_sequence=True)
        # print('zs', zs.shape)
        zs = self.norm(zs)
        xs = self.l2o(zs)
        return xs

class latentODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(latentODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = RNNEncoder(hidden_dim, hidden_dim, latent_dim)
        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

        self.in_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x, t, MAP=False):
        x = self.in_proj(x)
        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)
        # x_p = self.out_proj(x_p)
        return x_p, z, z_mean, z_log_var

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        seed_x = self.in_proj(seed_x)
        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        x_p = self.out_proj(x_p)
        return x_p