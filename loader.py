from typing import Dict
import torch
from vocab.vocab import Vocab
from vocab.vocab_preprocess import read_corpus

def read_data(args: Dict):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    return train_data, dev_data

def read_vocab(args: Dict):
    vocab = Vocab.load(args['--vocab'])
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    return vocab, vocab_mask

def read_model_params(args: Dict):
    train_batch_size = int(args['--batch-size'])
    N = int(args['--N'])
    d_model = int(args['--d_model'])
    d_ff = int(args['--d_ff'])
    h = int(args['--h'])
    dropout = float(args['--dropout'])

    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr = float(args['--lr'])

    return train_batch_size, N, d_model, d_ff, h, dropout, valid_niter, log_every, model_save_path, lr

