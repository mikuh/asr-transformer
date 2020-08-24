import os
from preprocess.extract_feature import extract_feature

from tensorflow.python.keras.utils.data_utils import get_file

wav_ch = {}
with open("/home/geb/PycharmProjects/asr/aishell_transcript_v0.8.txt", 'r', encoding='utf-8') as f:
    for line in f:
        a = line.strip().split()
        k = a[0]
        v = a[1:]
        wav_ch[k] = " ".join(v)

# wav_dir = "/home/geb/PycharmProjects/asr/S0002"


import glob
from tqdm import tqdm


def transfer_data(mode="train"):
    with open("/run/user/1000/gvfs/smb-share:server=jonas-dt.local,share=upload/asr-data/aishell-1-data/{}.tsv".format(mode), 'w',
              encoding='utf-8') as f:
        for filename in tqdm(glob.glob(
                '/run/user/1000/gvfs/smb-share:server=jonas-dt.local,share=upload/asr-data/aishell-1/*/{}/*/*.wav'.format(mode))):
            try:
                fbank_feat, id = extract_feature(filename)
            except:
                continue
            if id in wav_ch:
                x = " ".join(map(str, fbank_feat.flatten()))
                y = wav_ch[id]
                f.write(f"{x}\t{y}\n")
            else:
                print(id, "不存在")


# transfer_data("dev")
transfer_data("test")

# with open("/run/user/1000/gvfs/smb-share:server=jonas-dt.local,share=upload/asr-data/aishell-1")
#
#
# with open("../data/asr_raw_data_26dim.tsv", 'w', encoding='utf-8') as f:
#     for filename in os.listdir(wav_dir):
#         fbank_feat, id = extract_feature(os.path.join(wav_dir, filename))
#         if id in wav_ch:
#             x = " ".join(map(str, fbank_feat.flatten()))
#             y = wav_ch[id]
#
#         f.write(f"{x}\t{y}\n")
