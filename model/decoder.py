import torch.nn as nn
from model.attention import MultiHeadedAttention
from model.common import clones, LayerNorm, SublayerConnection, PositionwiseFeedForward

class DecoderLayer(nn.Module):
    "Decoder is made up of self-attn, src-attn, and feed-froward"

    def __init__(self, size: int,
                 self_attn: MultiHeadedAttention,
                 src_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        @param x        - shape (batch_size, sentence_len_dec, d_model)
        @param memory   - shape (batch_size, sentence_len_enc, d_model) # Note: `memory` is the final embedding in the top of the encoder stack.
        @param src_mask - shape (batch_size,                1, sentence_len_enc)
        @param tgt_mask - shape (batch_size, sentence_len_dec, sentence_len_dec) # Because we have to do subsequent masking.
        """
        "Follow Figure 1 for connections"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)



class Decoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        @param x        - shape (batch_size, sentence_len_dec, d_model)
        @param memory   - shape (batch_size, sentence_len_enc, d_model)
        @param src_mask - shape (batch_size,                1, sentence_len_enc)
        @param tgt_mask - shape (batch_size, sentence_len_dec, sentence_len_dec) # Because we have to do subsequent masking.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


