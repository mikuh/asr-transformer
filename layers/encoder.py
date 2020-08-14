import tensorflow as tf
from layers.attention import MultiHeadAttention
from layers.dense_einsum import DenseEinsum
from utils.activations import gelu


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,):
        super(Encoder, self).__init__()
        pass


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 intermediate_size,
                 num_heads,
                 dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 is_training=True,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        assert d_model % num_heads == 0
        head_size = d_model // num_heads

        self._mha = MultiHeadAttention(num_heads=num_heads,
                                       head_size=head_size,
                                       dropout_rate=0.0,
                                       attention_dropout_rate=attention_dropout_rate,
                                       is_training=is_training,
                                       **kwargs)

        self._add_norm = AddAndNorm(d_model=d_model,
                                    dropout_rate=dropout_rate,
                                    kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_initializer=bias_initializer,
                                    bias_regularizer=bias_regularizer,
                                    bias_constraint=bias_constraint)

        self._pwffn = PointWiseFFN(d_model=d_model,
                                   intermediate_size=intermediate_size,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=bias_regularizer,
                                   bias_constraint=bias_constraint)

        self._out_add_norm = AddAndNorm(d_model=d_model,
                                        dropout_rate=dropout_rate,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_constraint=kernel_constraint,
                                        bias_initializer=bias_initializer,
                                        bias_regularizer=bias_regularizer,
                                        bias_constraint=bias_constraint)

        self._dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        mha_output = self._mha(inputs)
        add_and_norm_output = self._add_norm(mha_output)
        pwffn_output = self._pwffn(add_and_norm_output)
        output = self._dropout(pwffn_output)
        return self._out_add_norm(output)


class AddAndNorm(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 kernel_constraint,
                 bias_initializer,
                 bias_regularizer,
                 bias_constraint):
        self._attention_output_dense = DenseEinsum(
            output_shape=d_model,
            num_summed_dimensions=2,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            name="transformer/self_attention_output")

        self._attention_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        # Use float32 in layernorm for numeric stability.
        # It is probably safe in mixed_float16, but we haven't validated this yet.
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="transformer/self_attention_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)

    def call(self, input):
        attention_output = self._attention_output_dense(input)
        attention_output = self._attention_dropout(attention_output, training=self.is_training)
        attention_output = self._attention_layer_norm(input + attention_output)
        return attention_output


class PointWiseFFN(tf.keras.layers.Layer):

    def __init__(self,
                 d_model,
                 intermediate_size,
                 kernel_initializer,
                 kernel_regularizer,
                 kernel_constraint,
                 bias_initializer,
                 bias_regularizer,
                 bias_constraint,
                 activation=gelu,
                 ):
        self._intermediate_dense = DenseEinsum(
            output_shape=intermediate_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            name="transformer/intermediate")

        self._output_dense = DenseEinsum(
            output_shape=d_model,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="transformer/output")

    def build(self, input_shape):
        pass

    def call(self, input):
        intermediate_output = self._intermediate_dense(input)
        layer_output = self._output_dense(intermediate_output)
        return layer_output
