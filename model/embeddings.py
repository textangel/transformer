import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self,x):
        """
        @param x: shape - (batch_size, sentence_len)
        :return: shape - (batch_size, sentence_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)


# Positional Embedding

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0., float(d_model), 2.) * (-1) * math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # If you have parameters in your transformer, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers using `self.register_buffer`
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: shape - (batch_size, sentence_len, d_model)
        :return: shape - (batch_size, sentence_len, d_model)
        """
        x = x + Variable(self.pe[:, :x.size(-2), :], requires_grad=False)
        return self.dropout(x)
