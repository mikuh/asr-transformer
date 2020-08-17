from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
import re

"/home/geb/PycharmProjects/asr/S0002/BAC009S0002W0122.wav"


def extract_feature(wav_file: str):
    rate, sig = wav.read(wav_file)
    fbank_feat = logfbank(sig, rate)
    _id = re.search(r'([^/]*)\.wav', wav_file)[1]
    return fbank_feat, _id




# print(fbank_feat.shape)
# fbank_feat = np.pad(fbank_feat, ((0, 1000 - fbank_feat.shape[0]), (0, 0)))

# print(fbank_feat.shape)

# A = np.array([[1, 2], [3, 4]])
#
# c = np.pad(A, ((0, 2), (0, 0)))
#
# print(c)
