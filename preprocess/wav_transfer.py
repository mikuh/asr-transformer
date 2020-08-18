import os
from preprocess.extract_feature import extract_feature

from tensorflow.python.keras.utils.data_utils import get_file

# wav_ch = {}
# with open("/home/geb/PycharmProjects/asr/aishell_transcript_v0.8.txt", 'r', encoding='utf-8') as f:
#     for line in f:
#         a = line.strip().split()
#         k = a[0]
#         v = a[1:]
#         wav_ch[k] = "".join(v)
#
# wav_dir = "/home/geb/PycharmProjects/asr/S0002"





# with open("../data/asr_raw_data_26dim.tsv", 'w', encoding='utf-8') as f:
#     for filename in os.listdir(wav_dir):
#         fbank_feat, id = extract_feature(os.path.join(wav_dir, filename))
#         if id in wav_ch:
#             x = " ".join(map(str, fbank_feat.flatten()))
#             y = wav_ch[id]
#
#         f.write(f"{x}\t{y}\n")

import gzip
import tarfile
import zipfile

file = "/home/geb/asr-data/AISHELL-ASR0009-OS1/resource_aishell.tgz"

with open(file, mode='rb') as fd:
    gzip_fd = gzip.GzipFile(fileobj=fd)
    tar = tarfile.open(gzip_fd.read())

    print(tar)