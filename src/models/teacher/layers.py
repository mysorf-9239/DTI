import math
import torch
import torch.nn as nn


def spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Sparse-dense matrix multiplication.
    A: sparse_coo_tensor shape (N, N)
    X: dense tensor shape (N, F)
    Returns: (N, F)
    """
    return torch.sparse.mm(A, X)


class GraphConvolution(nn.Module):
    """
    Basic GCN layer: A * (XW) (+ b)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)

    def forward(self, adj_sp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        support = x @ self.weight
        out = spmm(adj_sp, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCNStack(nn.Module):
    """
    Stack of 3 GCN layers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.g1 = GraphConvolution(dim, dim)
        self.g2 = GraphConvolution(dim, dim)
        self.g3 = GraphConvolution(dim, dim)

    def forward(self, adj: torch.Tensor, feat: torch.Tensor):
        h1 = torch.relu(self.g1(adj, feat))
        h2 = torch.relu(self.g2(adj, h1))
        h3 = torch.relu(self.g3(adj, h2))
        return h1, h2, h3
