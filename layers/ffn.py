import tensorflow as tf


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, intermediate_size, d_model, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.intermediate_size = intermediate_size
        self.d_model = d_model

        # (batch_size, seq_len, intermediate_size)
        self.dense1 = tf.keras.layers.Dense(intermediate_size, activation='relu')
        # (batch_size, seq_len, d_model)
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == '__main__':
    sample_ffn = PointWiseFeedForwardNetwork(3072, 768)
    x = tf.random.uniform((32, 1000, 768))
    print(sample_ffn(x).shape)
