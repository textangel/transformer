import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

def clones(module: nn.Module, N: int):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)"
    def __init__(self, d_model: int, eps=1e-16):
        """
        self.a_2 and self.b_2 are weights and biases which will apply across the embedding dimension (d_model)
        """
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Takes the mean and the standard deviation on the last dimension (d_model embedding dimension)
        for every batch and every position.
        @param x - shape ( * ,  d_model). x.shape[-1] must be d_model. The mean and std will be computed along this last dimension
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer: nn.Module):
        """
        @param x - shape (batch_size, sent_len, dim_model). The first dimension has size batch_size and the last has size dim_model.
        """
        "Apply the residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        @param x - shape (batch_size, sent_len, dim_model). The first dimension has size batch_size and the last has size dim_model.
        return shape: (batch_size, sent_len, dim_model)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))