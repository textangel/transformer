import torch.nn as nn
from model.attention import MultiHeadedAttention
from model.common import clones, LayerNorm, SublayerConnection, PositionwiseFeedForward
import torch

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self,
                 size: int,
                 self_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float):

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        @param x        - shape (batch_size, sentence_len_enc, d_model)
        @param mask     - shape (batch_size,                1, sentence_len_enc)
        """
        "Follow Figure 1 for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core Encoder is a stack of N layers"
    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        @param x        - shape (batch_size, sentence_len_enc, d_model)
        @param mask     - shape (batch_size,                1, sentence_len_enc)
        """
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
