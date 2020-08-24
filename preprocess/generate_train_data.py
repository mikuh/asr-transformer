import os
import numpy as np
import tensorflow as tf
import collections
import logging
from utils import tokenizer

logging.basicConfig(level=logging.INFO)


class InputExample(object):
    def __init__(self, guid, x, y=None):
        self.guid = guid
        self.x = x
        self.y = y


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 source_input,
                 source_mask,
                 target,
                 is_real_example=True):
        self.source_input = source_input
        self.source_mask = source_mask
        self.target = target
        self.is_real_example = is_real_example


class AsrProcessor(object):

    def get_train_examples(self, data_dir):
        lines = self._read_data(os.path.join(data_dir, "train.tsv"))
        return self._create_example(lines, "train")

    def get_dev_examples(self, data_dir):  # Development
        lines = self._read_data(os.path.join(data_dir, "dev.tsv"))
        return self._create_example(lines, "dev")

    def get_test_examples(self, data_dir):
        lines = self._read_data(os.path.join(data_dir, "test.tsv"))
        return self._create_example(lines, "test")

    def _create_example(self, lines, set_type):
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            yield InputExample(guid=guid, x=line[0], y=line[1])

    def _read_data(self, input_file):
        with tf.io.gfile.GFile(input_file, "r") as f:
            for line in f:
                yield line.strip().split("\t")


def convert_single_example(ex_index, example, max_source_length, max_target_length, tokenizer, feature_dim=26):
    input_feature = np.reshape(np.array(example.x.split(), dtype=np.float64), (-1, feature_dim))
    input_mask = [1 for _ in range(input_feature.shape[0])] + [0 for _ in
                                                               range(max_source_length - input_feature.shape[0])]
    if input_feature.shape[0] > max_source_length:
        return
    input_feature = np.pad(input_feature, ((0, max_source_length - input_feature.shape[0]), (0, 0)))
    target_text = example.y
    target_ids = []
    for word in target_text.split():
        if word.encode("utf-8").isalpha():
            target_ids.append(tokenizer.token2id(word))
        else:
            for ch in word:
                target_ids.append(tokenizer.token2id(ch))
    if len(target_ids) > max_target_length - 2:
        return
    target_ids = [tokenizer.vocab["[GO]"]] + target_ids + [tokenizer.vocab["[EOS]"]]
    for _ in range(max_target_length - len(target_ids)):
        target_ids.append(tokenizer.vocab["[PAD]"])

    assert input_feature.shape[0] == max_source_length
    assert len(input_mask) == max_source_length
    assert len(target_ids) == max_target_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("input_feature: %s" % " ".join([str(x) for x in input_feature[:5, :]]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("target_ids: %s" % " ".join([str(x) for x in target_ids]))

    feature = InputFeatures(
        source_input=input_feature.flatten(),
        source_mask=input_mask,
        target=target_ids,
        is_real_example=True
    )
    return feature


def file_based_convert_examples_to_features(examples, max_source_length, max_target_length, tokenizer, output_file,
                                            feature_dim=26):
    writer = tf.io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 500 == 0:
            logging.info("Writing example %d ." % (ex_index))
        feature = convert_single_example(ex_index, example, max_source_length, max_target_length, tokenizer,
                                         feature_dim)
        if feature is None:
            continue

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["source_input"] = create_float_feature(feature.source_input)
        features["source_mask"] = create_int_feature(feature.source_mask)
        features["target"] = create_int_feature(feature.target)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    asr_processor = AsrProcessor()

    data_dir = "/run/user/1000/gvfs/smb-share:server=jonas-dt.local,share=upload/asr-data/aishell-1-data"

    test_examples = asr_processor.get_test_examples(data_dir)

    train_examples = asr_processor.get_train_examples(data_dir)
    file_based_convert_examples_to_features(train_examples, 1000, 50, tokenizer, "../data/train.record0")

    # dev_examples = asr_processor.get_dev_examples(data_dir)
    # file_based_convert_examples_to_features(dev_examples, 1000, 50, tokenizer, "../data/dev.record0")
