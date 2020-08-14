import tensorflow as tf
from layers.dense_einsum import DenseEinsum


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 kernel_initializer: str = "glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 dropout_rate=0.0,
                 is_training=True,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._head_size = head_size
        self._dropout_rate = dropout_rate
        self._training = is_training

        self._query_dense = DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            name="attention/query")

        self._key_dense = DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            name="attention/key")

        self._value_dense = DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            name="attention/value")

        self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    def build(self, input_shape):
        self._query_dense.build(input_shape[0])
        self._key_dense.build(input_shape[1])
        self._value_dense.build(input_shape[1])
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        form_tensor = inputs[0]
        to_tensor = inputs[1]
        attention_mask = inputs[2] if len(inputs) == 3 else None

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        # `query_tensor` = [B, F, N ,H]
        query_tensor = self._query_dense(form_tensor)

        # `key_tensor` = [B, T, N, H]
        key_tensor = self._key_dense(to_tensor)

        # `value_tensor` = [B, T, N, H]
        value_tensor = self._value_dense(to_tensor)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)

        attention_scores = tf.multiply(attention_scores, 1.0 / tf.math.sqrt(float(self._head_size)))

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        if attention_mask is not None:
            attention_scores += self.create_padding_mask(attention_mask) * -1e9

        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = self._dropout(attention_probs, self._training)

        return tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), self.dtype)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
