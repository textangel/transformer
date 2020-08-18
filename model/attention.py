import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from .common import clones

# Attention
## An attention function can be described as mapping a query and a set of key-value pairs to an output,
## where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
## of the values, where the weight assigned to each value is computed by a compatibility function of the
## query with the corresponding key.
## We call our particular attention “Scaled Dot-Product Attention”. The input consists of queries and keys of dimension
## `d_k`, and values of dimension `d_v`. We compute the dot products of the query with all keys, divide each by
## \sqrt{d_k}, and apply a softmax function to obtain the weights on the values.

# In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix
# Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:
# $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

def attention(query, key, value, mask=None, dropout=None):
    #TODO: 1. Must duplicate the query in the case that we are using it with beam size
    """
    This is a general use attention function.

    @param query: shape - (batch_size,[beam_size], sent_len , d_model) - [note that sent_len can be src or tgt]
    @param key: shape - (batch_size, [beam_size], sent_len , d_model) - [note that sent_len can be src or tgt]
    @param value: shape - (batch_size, [beam_size], sent_len , d_model) - [note that sent_len can be src or tgt]
    @param mask: shape - (batch_size, 1 ,sent_len) - [note that sent_len can be src or tgt]

    If called on behalf of MultiHeadedAttention, the shapes are

    @param query: shape - (batch_size, [beam_size], h, sent_len,  d_k) - [note that sent_len can be src or tgt]
    @param key: shape - (batch_size,[beam_size], h, sent_len, d_k) - [note that sent_len can be src or tgt]
    @param value: shape - (batch_size, [beam_size],h, sent_len, d_k) - [note that sent_len can be src or tgt]
    @param mask: shape - (batch_size, 1, 1 , sent_len) - [note that sent_len can be src or tgt]

    In either case, only the last two dimensions are multiplied, the first or the first two dimensions are treated as batch.
    """
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    "The transpose here is necessary so that the matmul will go through"

    "scores has shape (batch_size, [h], sent_len,  sent_len)"
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    "p_attn has shape (batch_size, [h], sent_len_query,  sent_len_key)"
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    "Essentially, we have split the embedding space into 8 sections and computed an attention over each of them"
    "We have also split the value into 8 sections. For each of the sections, the post-attention embeddings is cast" \
    "in terms of that section in the value, with each section recpnstituted by a separete attention distrbution."
    "torch.matmul(p_attn, value) has shape (batch_size, h, sent_len_key, d_k) or (batch_size, sent_len_key, d_model). Recall sent_len_value = sent_len_key"
    return torch.matmul(p_attn, value), p_attn

# NB - Attention: The two most commonly used attention functions are **additive attention**, and **dot-product** (multiplicative) attention.
# Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{d_k}$.
# Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.
# While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient
# in practice, since it can be implemented using highly optimized matrix multiplication code.

# While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention
# without scaling for larger values of $d_k$. We suspect that for large values of $d_k$, the dot products grow large in magnitude,
# pushing the softmax function into regions where it has extremely small gradients.
# (To illustrate why the dot products get large, assume that the components of
# $q$ and $k$ are independent random variables with mean 0 variance 1. Then their dot product,
# $q \cdot k = \sum_{i=1}^{d_k}q_i k_i$ has mean 0 and variance $d_k$). To conteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}$

# Multi-Head Attention
# Multi-head attention allows the transformer to jointly attend to information from different representation
# subspaces at different positions. With a single attention head, averaging inhibits this.
# $\text{MultiHead}(Q,K,V) = Concat(head_1, ..., head_h) W^O$, where $head_i = Attention()QW^Q_i, KW^K_i, VW^V_i$

# Where the projections are the parameter matrices $W_^Q_i \in R^{d_{transformer} \times d_k}$,
# $W_i^K \in R^{d_{transformer} \times d_k}$, $W_i^V \in R^{d_{transformer} \times d_v}$, and $W^O \in R^{hd_v \times d_{transformer}}$.
# In this work we employ $h=8$ parallel attention layers or heads. For each of these we use $d_k = d_v = d_model / h = 64$.
# Due to the reduced dimensionality of each head, the total computational cost is similar to that of single-headed attention with
# full dimensionality.

# TODO Test if results are the same as original attention.

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in transformer size and number of heads"
        super(MultiHeadedAttention, self).__init__()
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

        assumes key == value
        """
        batch_size = query.size(0)
        if len(query.shape) == 3: query = query.unsqueeze(1)
        if len(key.shape) == 3: key = key.unsqueeze(1)
        if len(value.shape) == 3: value = value.unsqueeze(1)

        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1).unsqueeze(1)


        # 1) Do all the linear projections in batch from d_model => d x d_k
        # Note: view instantiates a tensor view
        # Note (1): The below code essentially cuts up each l(x) into h groups of dim d_k each,
        # And the attention is applied on the smaller groups, not on the original unsplit tensors.
        "l(x) does not change the shape. The view and transpose transforms the (batch_size, sentence_len, d_model)" \
        "tensor into shape (batch_size, h, sentence_len, d_k)"
        query, key, value = \
            [l(x).view(batch_size, x.shape[1], -1, self.h, self.d_k).transpose(-2,-3)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        "x has shape (batch_size, h, sent_len, d_k)"
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        # The below code essentially cuts up each l(x) into h groups of dim d_k each,
        # And the attention is applied on the smaller groups, not on the original unsplit tensors.
        # Note(2): After applying the attentions, the below code recomposes the embeddings of size
        # d_model (pieces back together).
        # Note: Contiguous() forces data to be contiguous by copying data if necessary.
        # And as for contiguous(..), it’s typically called because most cases view(...)
        # would throw an error if contiguous(..) isn’t called before.
        "x has shape (batch_size, h, sent_len, d_k) and is reshaped into shape (batch_size, sent_len, d_model) "
        x = x.transpose(-2, -3).contiguous() \
            .view(batch_size, x.shape[1], -1, self.h * self.d_k)

        if x.shape[1] == 1: x = x.squeeze(1)
        "return shape: (batch_size, sent_len, d_model)"
        return self.linears[-1](x)
