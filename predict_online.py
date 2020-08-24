import requests
from utils import tokenizer, parse_wav, generate_input
import tensorflow as tf
from preprocess.extract_feature import extract_feature
import pprint
import numpy as np

def pre_process(wav_file):
    log_bank = parse_wav(wav_file)
    input, mask, target = generate_input(log_bank)
    return input, mask, target


def post_process(predicted_ids):
    y_predict = u"".join([tokenizer.id2token(id) for id in predicted_ids])
    return y_predict


if __name__ == '__main__':
    wav_file = "/home/geb/PycharmProjects/asr/S0002/BAC009S0002W0137.wav"

    _input, _mask, _target = pre_process(wav_file)

    _tar_inp = _target + [4234]

    predicted_ids = []
    for i in range(49):
        tar_inp = _tar_inp
        resp = requests.post('http://localhost:8501/v1/models/asrtransfer:predict',
                                    json={"inputs": {"input_1": [_input.tolist()], "input_2": [_mask],
                                                     "input_3": [tar_inp]}})
        print(resp.json())
        predictions = np.array(resp.json()["outputs"][0])
        predictions = predictions[-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = int(np.argmax(predictions, axis=-1)[0])
        print(predicted_id)
        if predicted_id == 2:
            break
        predicted_ids.append(predicted_id)

        _tar_inp.append(predicted_id)


    # pprint.pprint(resp.json())
