import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from typing import Callable, List
from .attention import MultiHeadedAttention
from .common import PositionwiseFeedForward
from .embeddings import PositionalEncoding, Embeddings
from collections import namedtuple
import numpy as np
from torch.autograd import Variable
from vocab.vocab import Vocab

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class Generator(nn.Module):
    "Define standard linear and softmax generation step."
    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        :param x: shape - (batch_size, d_model). This one predicts one token at a time.
        :return: shape - (batch_size, vocab_size).
        """
        res = F.log_softmax(self.proj(x), dim=-1)
        return res

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Bsae for this and for many other models.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: Callable, tgt_embed: Callable, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src:      - shape (batch_size, sentence_len_enc)
        :param tgt:      - shape (batch_size, sentence_len_dec)
        :param src_mask: - shape (batch_size,                1, sentence_len_enc)
        :param tgt_mask: - shape (batch_size, sentence_len_dec, sentence_len_dec) # Because we have to do subsequent masking.
        :return: shape - (batch_size, sentence_len_dec, d_model)
        """
        "Take in and process masked src and target sequences"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        :param src:      - shape (batch_size, sentence_len_enc)
        :param src_mask: - shape (batch_size,                1, sentence_len_enc)
        :return: shape - (batch_size, sentence_len_enc, d_model)
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        :param memory:     (batch_size, sentence_len_enc, d_model)
        :param src_mask:   (batch_size,                1, sentence_len_enc)
        :param tgt:        (batch_size, sentence_len_dec)
        :param tgt_mask:   (batch_size, sentence_len_dec, sentence_len_dec) # Because we have to do subsequent masking.
        :return: shape - (batch_size, sentence_len_dec, d_model)
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



def subsequent_mask(size: int, batch_size=1):
    """
    :return: shape (batch_size, size, size) - every element in the batch is a triangular matrix
     with diagonal 1s, lower-triandle 1s, and upper-triangle 0s
    """
    "Mask out subsequent positions."
    attn_shape = (batch_size, size, size)
    # Note: Here np.triu is upper triabgle of an array
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a transformer from hyperparameters"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        Generator(d_model, tgt_vocab_size)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

### Use Only If Performing Translation together with CS224N vocab
#TODO UNTESTED
class TransfomerModel(nn.Module):
    def __init__(self, vocab: Vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, device='cpu'):
        super(TransfomerModel, self).__init__()
        self.vocab = vocab
        self.src_vocab_size = len(vocab.src)
        self.tgt_vocab_size = len(vocab.tgt)
        self.device = device
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        self.model = make_model(self.src_vocab_size, self.tgt_vocab_size,
                                N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout).to(device)

    @staticmethod
    def load(model_path: str):
        "Status: Tested"

        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        transformer_model = TransfomerModel(vocab=params['vocab'], **args)
        transformer_model.load_state_dict(params['state_dict'])
        return transformer_model

    def save(self, path: str):
        "Status: Tested"
        print(f"Save model params to {path}.")

        params = {
            'args': dict(N=self.N, d_model=self.d_model, d_ff=self.d_ff,
                         h=self.h, dropout=self.dropout, device=self.device),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    # This code predicts a translation using greedy decoding for simplicity.
    def greedy_decode(self, src, max_len=30):
        """
        :param src: shape (batch_size, sent_len)
        :return:
        """
        src = src.transpose(0,1)
        src_mask = (src != self.vocab.src.word2id['<pad>']).unsqueeze(-2) #TODO Untested
        start_symbol = self.vocab.src.word2id['<s>']
        "memory has shape (batch_size, sentence_len, d_model)"
        memory = self.model.encode(src, src_mask)
        batch_size = src.shape[0]
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = self.model.decode(memory, src_mask, Variable(ys).to(self.device),
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(self.device))
            prob = self.model.generator(out[:,-1,:])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1).type_as(src.data)], dim=1)
        return ys

    # Implement Batch Beam Search As Follows:
    #     For a Batch of Data
    #     - keep track of all live hypotheses in a batch ("the hypothesis tensor"). So if batch size is 4, and we keep track of beam size = 5
    #            We will keep track of 20 hypotheses at once. When generating new hypotheses, we generate top 5 from our 20
    #            (now the tensor has height 100), and then, for each of the 4 separate sentences,
    #            whittle the 25 corresponding entries back to 5.
    #     - We can store all the live entries simply as a large tensor. Alternately we can have a separate tensor for each
    #           sentence in the target, but this seems very suboptimal. Just keep a tensor of length batch_size * beam_size,
    #           and instead of taking the max value, take the topk values every step, and make sure to whittle back down to beam_size per target.
    #     - Have a map (index?) that tracks of the scores corresponding to the live entries for each target. Could also do this very simply
    #            by having an array where the index stores the score of the candidate sentence at that index in the hypothesis tensor.
    #

    def beam_search_decode(self, src: torch.Tensor, max_len: int = 45, beam_size: int = 5) -> List[List[Hypothesis]]:
        """
        An Implementation of Beam Search for the Transformer Model.
        Beam search is performed in a batched manner. Each example in a batch generates `beam_size` hypotheses.
        We return a list (len: batch_size) of list (len: beam_size) of Hypothesis, which contain our output decoded sentences
        and their scores.

        :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
        :param max_len: the maximum length to decode
        :param beam_size: the beam size to use
        :return completed_hypotheses: A List of length batch_size, each containing a List of beam_size Hypothesis objects.
            Hypothesis is a named Tuple, its first entry is "value" and is a List of strings which contains the translated word
            (one string is one word token). The second entry is "score" and it is the log-prob score for this translated sentence.

        Note: Below I note "4 bt", "5 beam_size" as the shapes of objects. 4, 5 are default values. Actual values may differ.
        """
        # 1. Setup
        batch_size = src.shape[1]
        start_symbol = self.vocab.src.word2id['<s>']
        end_symbol = self.vocab.src.word2id['</s>']

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        src = src.transpose(0,1)
        "src_mask has shape (batch_size, 1, sent_len)"
        src_mask = (src != self.vocab.src.word2id['<pad>']).unsqueeze(-2) #TODO Untested
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        model_encodings = self.model.encode(src, src_mask)

        # 1.2 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1,1), start_symbol, dtype=torch.long,
                                 device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.long,
                                 device=self.device)) for _ in range(batch_size)] # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len - 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]): break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i+1]] * cur_beam_size
                src_mask_l += [src_mask[i:i+1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            out = self.model.decode(model_encodings_cur, src_mask_cur, Variable(y_tm1).to(self.device),
                                Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device))
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            log_prob = self.model.generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1) \
                                          .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i])\
                                          .view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // len(self.vocab.tgt), top_cand_hyp_pos % len(self.vocab.tgt)

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [],[] # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat((hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id])))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.tgt.id2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0,-1)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                    value=[self.vocab.tgt.id2word[a.item()] for a in hypotheses[i][id][1:]],
                    score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print('completed_hypotheses', completed_hypotheses)
        return completed_hypotheses
