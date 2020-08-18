import tensorflow as tf
from models import Transformer
import matplotlib.pyplot as plt
from utils.data_utils import create_asr_transfer_dataset
import os
import time
import keras.backend as K

num_layers = 2
source_length = 1000
feature_dim = 26
target_length = 50
batch_size = 10
d_model = 512
dff = 2048
num_heads = 8
target_vocab_size = 26977
dropout_rate = 0.1
pe_input = 4096
pe_target = 512
log_steps = 1
model_dir = "results/asr-transfer/1/"


epochs = 1

train = False
export = True


# 将 Adam 优化器与自定义的学习速率调度程序（scheduler）配合使用

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,

        }
        return config


learning_rate = CustomSchedule(d_model)

# temp_learning_rate_schedule = CustomSchedule(d_model)
#
# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
#
# plt.show()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32)


def get_callbacks(model_dir):
    # custom_callback = keras_utils.TimeHistory(
    #     batch_size=train_batch_size,
    #     log_steps=log_steps,
    #     logdir=os.path.join(model_dir, 'logs'))

    summary_callback = tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'graph'), update_freq='batch')

    checkpoint_path = os.path.join(model_dir, 'checkpoint-{epoch:02d}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True)

    return [summary_callback, checkpoint_callback]


if __name__ == '__main__':

    with tf.distribute.OneDeviceStrategy("device:GPU:0").scope():
        if train:
            train_data, train_data_size = create_asr_transfer_dataset(
                "/home/geb/PycharmProjects/asr-transformer/data/test.record",
                source_length,
                feature_dim,
                target_length,
                batch_size,
                True)
            dev_data, dev_data_size = create_asr_transfer_dataset(
                "/home/geb/PycharmProjects/asr-transformer/data/test.record",
                source_length,
                feature_dim,
                target_length,
                batch_size,
                is_training=True)

            steps_per_epoch = train_data_size // batch_size
            eval_steps = dev_data_size // batch_size

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

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

            transformer.fit(
                train_data,
                validation_data=dev_data,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_steps=eval_steps,
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
            checkpoint_path = "results/asr-transfer/1/checkpoint-{:02d}".format(epochs)
            transformer.load_weights(checkpoint_path)
            transformer.predict((tf.ones([1, 1000, 26]), tf.ones([1, 1000]), tf.ones([1, 49])))

            tf.keras.models.save_model(transformer, saved_model_path, save_format='tf')
