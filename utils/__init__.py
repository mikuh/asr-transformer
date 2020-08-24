from utils.tf_utils import create_padding_mask, create_look_ahead_mask
import collections
import tensorflow as tf
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np


def parse_wav(file):
    rate, sig = wav.read(file)
    fbank_feat = logfbank(sig, rate)
    return fbank_feat


def generate_input(input_feature, max_source_length=1000):
    if input_feature.shape[0] > 1000:
        input_feature = input_feature[:1000, :]
    input_mask = [1 for _ in range(input_feature.shape[0])] + \
                 [0 for _ in range(max_source_length - input_feature.shape[0])]
    input_feature = np.pad(input_feature, ((0, max_source_length - input_feature.shape[0]), (0, 0)))
    target_input = [1]
    return input_feature, input_mask, target_input


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

if __name__ == '__main__':
    logfbank = parse_wav("/home/geb/PycharmProjects/asr/english.wav")
    input_feature, mask, target = generate_input(logfbank)
    print(input_feature.shape)
    print(mask)
    print(target)
