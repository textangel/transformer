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

    # This code predicts a translation using greedy decoding for simplicity.
    def beam_search_decode(self, src, max_len=45, beam_size = 5):
        """
        :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
        :return:
        """
        batch_size = src.shape[1]
        start_symbol = self.vocab.src.word2id['<s>']
        end_symbol = self.vocab.src.word2id['</s>']

        "src has shape (batch_size, sent_len)"
        src = src.transpose(0,1)
        "src_mask has shape (batch_size, 1, sent_len)"
        src_mask = (src != self.vocab.src.word2id['<pad>']).unsqueeze(-2) #TODO Untested
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        model_encodings = self.model.encode(src, src_mask)

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

        "List len batch_sz of shape (cur beam_sz, dec_sent_len), init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1,1), start_symbol, dtype=torch.long,
                                 device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.long,
                                 device=self.device)) for _ in range(batch_size)] # probs are log_probs must be init at 0.

        for iter in range(max_len - 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]): break
            cur_beam_sizes = []
            last_tokens = []
            model_encodings_l = []
            src_mask_l = []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]#[hypotheses[i][:,-1:]]
                model_encodings_l += [model_encodings[i:i+1]] * cur_beam_size
                src_mask_l += [src_mask[i:i+1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)# hypotheses[:,:,-1:].reshape(batch_size * cur_beam_size, 1)
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

            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                contiuating_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1) \
                                          .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i])\
                                          .view(-1)
                "shape (4 bt,5 beam)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (4 bt, 5 beam_size). Vals are between 0 and 50002 vocab_sz * 1 dec_sent_len. Note that dec_sent_len may be >1 in some applications"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores_i, k=live_hyp_num_i)
                "shape (4 bt, 5 beam_size). prev_hyp_ids vals are 0 <= val < beam_size. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // len(self.vocab.tgt), top_cand_hyp_pos % len(self.vocab.tgt)

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

                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0,-1)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i = None
                    hyp_scores_i = None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                    value=[self.vocab.tgt.id2word[a.item()] for a in hypotheses[i][id][1:]],
                    score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        print('completed_hypotheses', completed_hypotheses)
        return completed_hypotheses

        #         # candidate_probs_all, candidate_ix_all = torch.topk(log_prob, beam_size, dim=1)
        #         ys = torch.cat([ys, candidate_ix_all.unsqueeze(-1)], dim = -1)
        #         continue
        #     else:
        #         # # INSERT CODE FOR ITERATIONS 2...
        #         "shape (batch * beam, decoded_len_so_far)"
        #         ys_orig = ys
        #         # ys = ys.view(batch_size * beam_size, -1)
        #         out = self.model.decode(model_encodings, src_mask, Variable(ys).to(self.device),
        #                                 Variable(subsequent_mask(ys.size(-1)).type_as(src.data)).to(self.device))
        #         # "log_prob is prob of this particular word given the previous. " \
        #         # "Log_probs are supposed to be added. This produces the full conditional prob (log version)" \
        #         # "of all the tokens starting from the first."
        #         log_prob = self.model.generator(out[:,:,-1,:])
        #         k_probs, k_next_word_ixs = torch.topk(log_prob, beam_size, dim=-1)
        #         "shape (4 bt,5 beam, 5 beam)"
        #         candidate_probs_all = k_probs + current_probs
        #         "shape (4 bt, 25 beam * beam)"
        #         candidate_probs_all = candidate_probs_all.view(-1, beam_size * beam_size)
        #
        #         topkprob_values, topkprob_ix = torch.topk(candidate_probs_all, 5, -1)
        #
        #         seq_len = ys.size(-1)
        #
        #         ix_in_lineage = torch.cat([(topkprob_ix // 5).unsqueeze(-1)] * seq_len, dim=-1)
        #         ix_in_newtoken = torch.cat([(topkprob_ix // 5).unsqueeze(1), (topkprob_ix % 5).unsqueeze(1)], dim=1)
        #         ixa=ix_in_newtoken
        #         ixx = topkprob_ix // 5
        #         ixy = topkprob_ix % 5
        #
        #         lineage = ys.gather(1, ix_in_lineage) # This works
        #
        #         k_next_word_ixs.gather(1,ix_in_newtoken) # TODO this doesnt work, fix
        #
        #         topkprob_ix // beam_size
        #         topkprob_ix % beam_size
        #         current_probs = topkprob_values.unsqueeze(1)
        #         # currnet_lineage =
        #
        #         ys.gather(1,prior_lineage_ix)
        #         # "shape (4 bt,5 beam)"
        #         # ys = torch.cat([ys, next_word.unsqueeze(1).type_as(src.data)], dim=1)
        #
        #         # candidate_ix_all_old = candidate_ix_all
        #         # candidate_ix_all = candidate_ix_all.view(-1, beam_size * beam_size)
        #
        #         # shape (4 bt, 5 beam)
        #         prior_lineage_ix = topkprob_ix // beam_size #TODO UNTESTED
        #         current_candidate_ix.gather(1,prior_lineage_ix) #TODO UNTESTED
        #         # full_lineage += current_candidate_ix
        #         ys = torch.cat([ys, current_candidate_ix.unsqueeze(1).type_as(src.data)], dim=1)
        #         "shape (4 bt,5 beam)"
        #         current_probs = candidate_probs_all.gather(1, topkprob_ix) # also = topkprob_values but whatever
        #         "shape (4 bt,5 beam)"
        #         current_candidate_ix = candidate_ix_all.gather(1, topkprob_ix)
        #
        #         "When you do this view you lose the dependency between the new token and the token it came from (dim -2)"
        #         "You will be able to get the top 5 of the 25 tokens in terms of log_prob, but you also have to " \
        #         "keep track of which of the previous dimensions (corresponding to previous tokens) it came from"
        #
        # return ys



    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)]) #TODO
        src_encodings_att_linear = self.att_projection(src_encodings)             #TODO

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)            #TODO

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,                     #TODO
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,                          #TODO
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses