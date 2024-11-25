# copied from https://github.com/andrejmiscic/gcn-pytorch/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor):
        x = self.__transform_to_chebyshev(x, laplacian)
        x = self.linear(x)
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x#.unsqueeze(2)
        x0 = x

        if self.K > 1:
            # x1 = torch.sparse.mm(laplacian, x0)
            x1 = torch.einsum("bnm,btmc->btnc", laplacian, x0)#torch.bmm(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1), -1)
            assert torch.isnan(x1).sum() == 0, f"gcn0 {torch.isnan(x1).sum()} {x.min()} {x.max()} {laplacian.min()} {laplacian.max()} {laplacian}"
            for _ in range(2, self.K):
                # x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
                x2 = 2 * torch.einsum("bnm,btmc->btnc", laplacian, x1) - x0
                assert torch.isnan(x2).sum() == 0, f"gcn{_} {torch.isnan(x2).sum()}"
                cheb_x = torch.cat((cheb_x, x2), -1)
                x0, x1 = x1, x2

        # cheb_x = cheb_x.reshape([x.shape[0], -1])
        return cheb_x


class ChebNetGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, dropout=0.1, residual=False, k=2):
        super(ChebNetGCN, self).__init__()

        self.dropout = dropout
        self.residual = residual

        self.input_conv = ChebNetConv(input_size, hidden_size, k)
        self.hidden_convs = nn.ModuleList([ChebNetConv(hidden_size, hidden_size, k) for _ in range(num_hidden_layers)])
        self.output_conv = ChebNetConv(hidden_size, output_size, k)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor, labels: torch.Tensor = None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_conv(x, laplacian))
        for conv in self.hidden_convs:
            if self.residual:
                x = F.relu(conv(x, laplacian)) + x
            else:
                x = F.relu(conv(x, laplacian))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_conv(x, laplacian)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss