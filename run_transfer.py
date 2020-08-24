import tensorflow as tf
from models import Transformer
import matplotlib.pyplot as plt
from utils.data_utils import create_asr_transfer_dataset
import os
import time
from utils import tokenizer
from utils import optimization
from utils import distribution_utils
import keras.backend as K

num_layers = 2
source_length = 1000
feature_dim = 26
target_length = 50
batch_size = 32
d_model = 512
dff = 2048
num_heads = 8
target_vocab_size = 4337
dropout_rate = 0.1
pe_input = 4096
pe_target = 512
log_steps = 1
model_dir = "results/asr-transfer/2/"
learning_rate = 2e-5
epochs = 150

train = False
export = False
predict = True

import numpy as np

np.set_printoptions(threshold=np.inf)


def get_optimizer(initial_lr, steps_per_epoch, epochs, warmup_steps, use_float16=False):
    optimizer = optimization.create_optimizer(initial_lr, steps_per_epoch * epochs, warmup_steps)
    return optimizer


# 将 Adam 优化器与自定义的学习速率调度程序（scheduler）配合使用

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=40000):
#         super(CustomSchedule, self).__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
#     def get_config(self):
#         config = {
#             'd_model': self.d_model,
#             'warmup_steps': self.warmup_steps,
#
#         }
#         return config
#
#
# learning_rate = CustomSchedule(d_model)
#
# temp_learning_rate_schedule = CustomSchedule(d_model)
#
# plt.plot(temp_learning_rate_schedule(tf.range(4000000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
#
# plt.show()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    # train_accuracy.update_state(tf.reshape(real, (batch_size, -1, 1)), pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def metric_fn():
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def accuracy(real, pred):
        sample_weight = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), tf.float32)
        train_accuracy.update_state(real, pred, sample_weight=sample_weight)
        return train_accuracy.result()

    return accuracy


def get_callbacks(model_dir):
    # custom_callback = keras_utils.TimeHistory(
    #     batch_size=train_batch_size,
    #     log_steps=log_steps,
    #     logdir=os.path.join(model_dir, 'logs'))

    summary_callback = tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'graph'), update_freq='batch')

    checkpoint_path = os.path.join(model_dir, 'checkpoint-{epoch:02d}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, save_freq=batch_size * 100)

    return [summary_callback, checkpoint_callback]


if __name__ == '__main__':

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy='one_device',
        num_gpus=1,
        tpu_address=False)

    with strategy.scope():
        if train:
            train_data, train_data_size = create_asr_transfer_dataset(
                "/home/geb/PycharmProjects/asr-transformer/data/train.record0",
                source_length,
                feature_dim,
                target_length,
                batch_size,
                True)
            dev_data, dev_data_size = create_asr_transfer_dataset(
                "/home/geb/PycharmProjects/asr-transformer/data/dev.record0",
                source_length,
                feature_dim,
                target_length,
                batch_size,
                is_training=False)

            steps_per_epoch = train_data_size // batch_size
            warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
            eval_steps = dev_data_size // batch_size * 10

            # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            optimizer = get_optimizer(learning_rate, steps_per_epoch, epochs, warmup_steps)

            callbacks = get_callbacks(model_dir)

            transformer = Transformer(num_layers,
                                      d_model,
                                      num_heads,
                                      dff,
                                      target_vocab_size,
                                      pe_input=pe_input,
                                      pe_target=pe_target,
                                      rate=dropout_rate,
                                      training=train)

            transformer.compile(optimizer=optimizer, loss=loss_function, metrics=[metric_fn()])

            checkpoint_path = "results/asr-transfer/2/checkpoint-{:02d}".format(37)
            transformer.load_weights(checkpoint_path)

            transformer.fit(
                train_data,
                # validation_data=dev_data,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                # validation_steps=eval_steps,
                callbacks=callbacks)

        elif export:
            transformer = Transformer(num_layers,
                                      d_model,
                                      num_heads,
                                      dff,
                                      target_vocab_size,
                                      pe_input=pe_input,
                                      pe_target=pe_target,
                                      rate=dropout_rate,
                                      training=False)
            saved_model_path = "saved_models/{}".format(int(time.time()))
            checkpoint_path = "results/asr-transfer/2/checkpoint-{:02d}".format(37)
            transformer.load_weights(checkpoint_path)
            transformer.predict((tf.ones([1, 1000, 26]), tf.ones([1, 1000]), tf.ones([1, 49])))

            tf.keras.models.save_model(transformer, saved_model_path, save_format='tf')
        elif predict:
            transformer = Transformer(num_layers,
                                      d_model,
                                      num_heads,
                                      dff,
                                      target_vocab_size,
                                      pe_input=pe_input,
                                      pe_target=pe_target,
                                      rate=dropout_rate,
                                      training=False)
            saved_model_path = "saved_models/{}".format(int(time.time()))
            checkpoint_path = "results/asr-transfer/2/checkpoint-{:02d}".format(37)
            transformer.load_weights(checkpoint_path)

            # a = transformer.predict((tf.ones([1, 1000, 26]), tf.ones([1, 1000]), tf.ones([1, 1])))
            # print(a.shape)

            import numpy as np
            import Levenshtein

            total = 0
            right = 0

            with open("data/test.tsv", 'r', encoding='utf-8') as f:
                for a in f:

                    a = a.split("\t")
                    y_true = u"".join(a[-1].split())
                    inp = np.reshape(np.array(a[0].split(), dtype=np.float64), (-1, 26))
                    print(inp.shape)
                    mask = np.array([[1 if i < inp.shape[0] else 0 for i in range(1000)]])
                    inp = np.array([np.pad(inp, ((0, 1000 - inp.shape[0]), (0, 0)))])
                    print(inp.shape)
                    _tar_inp = [1]

                    predicted_ids = []
                    for i in range(50):
                        tar_inp = np.array([_tar_inp])
                        print(inp.shape, mask.shape, tar_inp.shape)
                        predictions = transformer.predict((inp, mask, tar_inp))
                        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
                        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32).numpy()[0][0]
                        if predicted_id == 2:
                            break

                        predicted_ids.append(predicted_id)

                        _tar_inp.append(predicted_id)

                    y_predict = u"".join([tokenizer.id2token(id) for id in predicted_ids])

                    print(y_true)
                    print(y_predict)
                    diff = Levenshtein.distance(y_true, y_predict)
                    print(diff)
                    right += len(y_true) - diff
                    total += len(y_true)

                    print("current accuracy:", right / total)
            print("Accuracy:", right / total)
