import tensorflow as tf
from layers.dense_einsum import DenseEinsum
from utils.activations import gelu


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
            name="encoder/intermediate")

        self._output_dense = DenseEinsum(
            output_shape=d_model,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name="encoder/output")

    def call(self, input):
        intermediate_output = self._intermediate_dense(input)
        layer_output = self._output_dense(intermediate_output)
        return layer_output
