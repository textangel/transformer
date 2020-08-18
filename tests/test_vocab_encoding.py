import unittest
import torch
from vocab.vocab import Vocab, text_to_tensor
from vocab.vocab_preprocess import read_corpus
#
# # Test 1
# class TestReadData:
#     pass # This works fine

class TestVocabToTensor(unittest.TestCase):
    def setUp(self):
        train_src = "en_es_data/train_tiny.en"
        train_tgt = "en_es_data/train_tiny.es"
        vocab_path = "vocab/vocab_tiny_q1.json"
        self.train_data_src = read_corpus(train_src, source='src')
        self.train_data_tgt = read_corpus(train_tgt, source='tgt')
        self.vocab = Vocab.load(vocab_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @unittest.skip("Passed")
    def test_most_words_in_vocab(self):
        """
        Make sure that most of our encoded words are not <pad> or <unk>
        """
        tensor_src, tensor_tgt = text_to_tensor(self.train_data_src, self.train_data_tgt, self.vocab, self.device)
        num_words_src = len(set([l for row in tensor_src.numpy() for l in row]))
        num_words_tgt = len(set([l for row in tensor_tgt.numpy() for l in row]))
        print("num_words_src", num_words_src)
        print("num_words_tgt", num_words_tgt)
        assert num_words_src > len(self.vocab.src) / 2
        assert num_words_tgt > len(self.vocab.tgt) / 2

    @unittest.skip("Passed")
    def test_VocabToTensor(self):
        tensor_result = text_to_tensor(self.train_data_src, self.train_data_tgt, self.vocab, self.device)
        for i in tensor_result:
            print(i)

        self.assertEqual(len(tensor_result), 2)
