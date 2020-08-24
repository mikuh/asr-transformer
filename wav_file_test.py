from pydub import AudioSegment
from models import Transformer
from utils import tokenizer, parse_wav, generate_input
import tensorflow as tf
import numpy as np

transformer = Transformer(2,
                          512,
                          8,
                          2048,
                          4337,
                          pe_input=4096,
                          pe_target=512,
                          rate=0.1,
                          training=False)
checkpoint_path = "results/asr-transfer/2/checkpoint-{:02d}".format(37)
transformer.load_weights(checkpoint_path)


def pre_process(wav_file):
    log_bank = parse_wav(wav_file)
    input, mask, target = generate_input(log_bank)
    return input, mask, target


def post_process(predicted_ids):
    y_predict = u"".join([tokenizer.id2token(id) for id in predicted_ids])
    return y_predict


def trans_mp3_to_wav(filepath):
    song = AudioSegment.from_mp3(filepath)
    target_name = filepath.split(".")[0] + ".wav"
    song.export(target_name, format="wav")



    # return msg.text
wav_file = "/home/geb/PycharmProjects/asr-transformer/data/A2_1.wav"
inp, mask, target = pre_process(wav_file)
inp = np.array([inp])
mask = np.array([mask])
_tar_inp = target
predicted_ids = []
for i in range(49):
    tar_inp = np.array([_tar_inp])
    predictions = transformer.predict((inp, mask, tar_inp))
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32).numpy()[0][0]
    if predicted_id == 2:
        break

    predicted_ids.append(predicted_id)

    _tar_inp.append(predicted_id)

y_predict = u"".join([tokenizer.id2token(id) for id in predicted_ids])
print(y_predict)


