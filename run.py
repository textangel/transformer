#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py: Run Script for Transformer Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Zhengyuan Ma <zhengm@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --log-every=<int>                       log every [default: 10]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --max-epoch=<int>                       max epoch [default: 3000]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 200]
    --dropout=<float>                       dropout [default: 0.3]
    --N=<int>                               transformer layers [default: 2]
    --d_model=<int>                         model embedding size [default: 128]
    --d_ff=<int>                            transformer feedforward embedding size [default: 1024]
    --h=<int>                               transformer multi-head attention num of heads [default: 4]
    --beam_size=<int>                       Beam Size for Beam Search [default: 8]
"""

"""
Step 1:
1.1 Generate the vocab from src and target sentences > vocab.json using:
python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json --size=50000 --freq-cutoff=2
1.2 You can then directly read in this vocab file using `vocab = Vocab.load(path)`
"""
from functools import partial
from collections import namedtuple
from typing import Dict, List
from docopt import docopt
import numpy as np
import time
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu

from loader import read_data, read_vocab, read_model_params
from model.model import TransfomerModel, Generator
from training.opt import NoamOpt
from training.batch import Batch
from training.label_smooth import LabelSmoothing
from vocab.vocab import text_to_tensor
from vocab.vocab_preprocess import batch_iter, read_corpus

#Probably tested
def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[List[str]]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp for hyp in hypotheses])
    return bleu_score


# Loss Computation
class SimpleLossCompute:
    """
    A simple loss compute and train function. We decode the output text one char position at a time.
    """
    def __init__(self, generator: Generator, criterion, optimizer, train=True):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer
        self.training = train

    # Here we generate the output text one column (word) at a time
    def __call__(self, out, y, norm):
        out_grad = []
        total_loss = 0
        for i in range(out.size(1)):
            out_column = Variable(out[:, i].data, requires_grad=True)
            "shape (batch_size, vocab_sz)"
            y_pred = self.generator(out_column)
            "shape (batch_size, )"
            y_actual = y[:, i].data
            loss = self.criterion(y_pred, y_actual) / norm
            total_loss += float(loss.cpu().data.item())
            if self.training:
                loss.backward()
                out_grad.append(out_column.grad.data.clone())
        
        if self.training:
            out_grad = torch.stack(out_grad, dim=1)
            out.backward(gradient=out_grad)

        self.optimizer.step()
        self.optimizer.optimizer.zero_grad()
        # Note: Since the backward() function accumulates gradients, and you don’t want to mix up gradients between minibatches,
        # you have to zero them out at the start of a new minibatch. This is exactly like how a general (additive) accumulator
        # variable is initialized to 0 in code.
        return total_loss * norm


def run_dev_session(model, dev_data, vocab, loss_compute, batch_size=32, device=torch.device('cpu')):
    """ Evaluate perplexity on dev sentences
    @param model (Transformer): Transformer Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()
    cum_loss = cum_tgt_words = cum_examples = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
      for train_iter, batch_dev_sents in enumerate(batch_iter(dev_data, batch_size=batch_size)):
          loss, batch_size, ntokens = train_step(model, batch_dev_sents, vocab, loss_compute, device=device)
          cum_loss += loss
          cum_examples += batch_size
          cum_tgt_words += ntokens
      dev_loss = cum_loss / cum_examples
      ppl = np.exp(cum_loss / cum_tgt_words)
      
    if was_training:
        model.train()

    return dev_loss, ppl

def train_step(model, batch_sents_data, vocab, loss_compute, device):
    # Tested
    src_sents, tgt_sents = batch_sents_data
    batch_size = len(src_sents)
    tensor_src, tensor_tgt = text_to_tensor(src_sents, tgt_sents, vocab, device)

    batch = Batch(tensor_src.transpose(0,1), tensor_tgt.transpose(0,1), vocab.tgt.word2id['<pad>'])
    out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
    loss = loss_compute(out, batch.trg_y, batch.ntokens)
    return loss.cpu(), batch_size, batch.ntokens.cpu().numpy()

def nllLoss(pad_token, y_pred, y):
    """
    y_pred (batch_size, vocab_sz) - is log softmax prob of outputs obtained by calling log_softmax on output embeddings.
    y (batch_size, ) - is ground truth - the words actually used.
    :return:
    """
    "shape (batch_size, 1) "
    y_reshaped = y.unsqueeze(-1)
    target_masks = (y_reshaped != pad_token).float()
    target_gold_words_log_prob = torch.gather(y_pred, index=y_reshaped, dim=-1) * target_masks
    loss = -1 * target_gold_words_log_prob.sum()
    return loss

def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    :param args:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('use device: %s' % device)
    train_data, dev_data = read_data(args)
    vocab, vocab_mask = read_vocab(args)
    train_batch_size, N, d_model, d_ff, h, dropout, valid_niter, log_every, model_save_path, lr = read_model_params(args)

    transformer_model = TransfomerModel(vocab, N, d_model, d_ff, h, dropout, device)
    model = transformer_model.model
    optimizer = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9))

    # criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(size=len(vocab.tgt.word2id), padding_idx=vocab.tgt.word2id['<pad>'], smoothing=0.0)
    criterion = LabelSmoothing(size=len(vocab.tgt.word2id), padding_idx=vocab.tgt.word2id['<pad>'],
                               smoothing=0.001)
    # criterion = partial(nllLoss, vocab.src.word2id['<pad>'])
    loss_compute_train = SimpleLossCompute(model.generator, criterion, optimizer)
    loss_compute_dev = SimpleLossCompute(model.generator, criterion, optimizer, train=False)

    train_time = start_time = time.time()
    patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    num_trial = cum_exmaples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []

    print('begin Maximum Likelihood Training')
    while True:
        epoch += 1
        for train_iter, batch_sents in enumerate(batch_iter(train_data, batch_size=train_batch_size, shuffle=True)):

            loss, batch_size, n_tokens = train_step(model, batch_sents, vocab, loss_compute_train, device)
            report_loss += loss
            cum_loss += loss
            cum_exmaples += batch_size
            report_examples += batch_size
            report_tgt_words += n_tokens
            cum_tgt_words += n_tokens

            if train_iter % log_every == 0:
                elapsed = time.time() - start_time
                elapsed_since_last = time.time() - train_time
                print(f"epoch {epoch}, iter {train_iter}, avg loss {report_loss / report_examples: .3f}, "
                      f"avg ppl {np.exp(report_loss / report_tgt_words): .3f}, cum examples {cum_exmaples}, "
                      f"speed {report_tgt_words/ elapsed_since_last: .3f} w/s, elapsed time {elapsed: .3f} s, lr= {optimizer._rate}")
                train_time = time.time()
                report_tgt_words = report_loss = report_examples = 0.

            if train_iter % valid_niter == 0:
                print(f"epoch {epoch}, iter {train_iter}, cum. loss {cum_loss/cum_exmaples}, "
                      f"cum ppl {np.exp(cum_loss / cum_tgt_words)}, cum exmples {cum_exmaples}, lr= {optimizer._rate}")
                cum_loss = cum_exmaples = cum_tgt_words = 0.
                valid_num += 1
                print("begin validation ...")

                dev_loss, dev_ppl = run_dev_session(model, dev_data, vocab, loss_compute_dev, batch_size=32, device=device)
                print(f'validation: iter {train_iter}, dev. loss {dev_loss}, dev. ppl {dev_ppl}')

                valid_metric = -dev_ppl
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
                if is_better:
                    patience = 0
                    print(f'save currently the best model to {model_save_path}')
                    transformer_model.save(model_save_path)
                    torch.save(optimizer.optimizer.state_dict(), model_save_path+".optim")
                elif patience < int(args['--patience']):
                    patience += 1
                    print(f'hit patience {patience}')
                    if patience == int(args['--patience']):
                        num_trial += 1
                        print(f"hit #{num_trial} trial")
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!')
                            exit(0)
                if epoch == int(args['--max-epoch']):
                    print('reached max number of epochs!')
                    exit(0)

def decode(args: Dict[str, str], max_batch_size=512, mode='greedy'):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("load model from {}".format(args['MODEL_PATH']))
    transformer_model = TransfomerModel.load(args['MODEL_PATH']).to(device)
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']))
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    tensor_src = transformer_model.vocab.src.to_input_tensor(test_data_src, device)
    num_examples = len(test_data_src)
    bs = max_batch_size
    hypotheses = []
    for i in range(0, num_examples, max_batch_size):
        tensor_src_i = tensor_src[:,i:i+bs]
        if mode == 'greedy':
            hypotheses_i = transformer_model.greedy_decode(tensor_src_i)
            hypotheses_i = hypotheses_i.cpu().detach().numpy()
            hypotheses_i = [[transformer_model.vocab.tgt.id2word[w] for w in sent] for sent in hypotheses_i]
            hypotheses += hypotheses_i
        elif mode == 'beam':
            beam_size = int(args.get('--beam_size', 8))
            hypotheses_batch = transformer_model.beam_search_decode(tensor_src_i, beam_size=beam_size)
            hypotheses += [hyp_i[0].value for hyp_i in hypotheses_batch]
        print(f"Decoded batches {i} - {i+bs}: {num_examples - i - bs} more to go!")
    print(hypotheses)

    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']))
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')
        # top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100))

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            hyp_sent = ' '.join(hyps)
            f.write(hyp_sent + '\n')


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args,mode='greedy')
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()

    # TODO: Potential issues:
    """ 
    1. CELoss() outputs very small numbers, may not be what we want. Check out label smoothing? √
    2. Need to see if we can do decoding all at once, instead of one word at a time. (Yes we can. When training. Look into this.)
    3. Model training may fail for some other reason.
    4. Saving Optimizer state (in training/opt.py) not yet implemented
    
    """