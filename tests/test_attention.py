import unittest
import torch
from model.attention import MultiHeadedAttention, attention
import torch.nn as nn
from model.common import clones
from model.model import subsequent_mask


class TestVocabToTensor(unittest.TestCase):
    def setUp(self):
        pass

    def test_multi_head_attention(self):
        q = k = v = torch.randn(4,7,128)
        k2 = v2 = torch.randn(4, 12, 128)
        mask = subsequent_mask(7)
        mask2 = torch.tensor([[1] * 11 + [0] * 1,
                 [1] * 8 + [0] * 4,
                 [1] * 10 + [0] * 2,
                 [1] * 3 + [0] * 9]).unsqueeze(1)
        print("mask.shape", mask.shape)
        torch.manual_seed(24)
        mha = MultiHeadedAttention(8,128,0)
        torch.manual_seed(24)
        omha = OriginalMultiHeadedAttention(8, 128, 0)
        assert (mha(q,k,v) == omha(q,k,v)).all()
        assert (mha(q, k2, v2) == omha(q, k2, v2)).all()
        assert (mha(q, k, v, mask) == omha(q, k, v, mask)).all()
        assert (mha(q, k2, v2, mask2) == omha(q, k2, v2, mask2)).all()



class OriginalMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in transformer size and number of heads"
        super(OriginalMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume here that d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value - shape (batch_size, sentence_len_enc, d_model) - for encoder values
                          - shape (batch_size, sentence_len_dec, d_model) - for decoder values

        mask - shape (batch_size,            1, sentence_len) for encoder mask
             - shape (batch_size, sentence_len, sentence_len) for decoder mask
        """
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        "tensor into shape (batch_size, h, sentence_len, d_k)"
        query, key, value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        "x has shape (batch_size, h, sent_len, d_k)"
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)
        "return shape: (batch_size, sent_len, d_model)"
        return self.linears[-1](x)