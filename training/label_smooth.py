import torch.nn as nn
import torch
from torch.autograd import Variable

# Regularization
## Label Smoothing
# During training, we employed label smoothing of value $\epsilon_{ls} = 0.1.
# This hurts perplexity, as the transformer learns to be more unsure, but improves accuracy and BLEU score.
# We implement label smoothing using the KL div loss. Instead of using a one-hot target distribution,
# we create a distribution that has confidence of the correct word and the rest of the smoothing mass
# distributed throughout the vocabulary.

class LabelSmoothing(nn.Module):
    "Implement Label Smoothing"
    def __init__(self, size, padding_idx, smoothing=0.0, criterion=None):
        super(LabelSmoothing, self).__init__()
        if criterion is None:
            criterion = nn.KLDivLoss(size_average=False)
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/ (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
