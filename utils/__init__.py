from utils.tf_utils import create_padding_mask, create_look_ahead_mask
import collections
import tensorflow as tf


class Tokenizer(object):
    vocab = collections.OrderedDict()
    a = []

    def __init__(self):
        self.load_vocab("/home/geb/PycharmProjects/asr-transformer/vocab_file/char.txt")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""

        index = 0
        with tf.io.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                self.vocab[token] = index
                self.a.append(token)
                index += 1

    def token2id(self, token):
        return self.vocab.get(token, self.vocab["[UNK]"])

    def id2token(self, id):
        return self.a[id]


tokenizer = Tokenizer()
