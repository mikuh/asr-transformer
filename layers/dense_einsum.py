import tensorflow as tf

_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]


class DenseEinsum(tf.keras.layers.Layer):

    def __init__(self,
                 output_shape,
                 num_summed_dimensions=1,
                 activation=None,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseEinsum, self).__init__(**kwargs)
        self._output_shape = output_shape if isinstance(output_shape, (list, tuple)) else (output_shape,)
        self._activation = tf.keras.activations.get(activation)
        self._num_summed_dimensions = num_summed_dimensions
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._use_bias = use_bias
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._einsum_string = None

    def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
        input_str = ""
        kernel_str = ""
        output_str = ""
        letter_offset = 0

        for i in range(free_input_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            output_str += char

        letter_offset += free_input_dims
        for i in range(bound_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            kernel_str += char

        letter_offset += bound_dims
        for i in range(output_dims):
            char = _CHR_IDX[i + letter_offset]
            kernel_str += char
            output_str += char

        return input_str + "," + kernel_str + "->" + output_str

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_rank = input_shape.rank
        free_input_dims = input_rank - self._num_summed_dimensions
        output_dims = len(self._output_shape)

        self._einsum_string = self._build_einsum_string(free_input_dims,
                                                        self._num_summed_dimensions,
                                                        output_dims)
        self._kernel_shape = input_shape[free_input_dims:].concatenate(self._output_shape)

        self._kernel = self.add_weight(
            name="kernel",
            shape=self._kernel_shape,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )

        self._bias = None
        if self._use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=self._output_shape,
                initializer=self._bias_initializer,
                regularizer=self._bias_regularizer,
                constraint=self._bias_constraint,
                dtype=self.dtype,
                trainable=True
            )

        super(DenseEinsum, self).build(input_shape)

    def call(self, inputs):
        ret = tf.einsum(self._einsum_string, inputs, self._kernel)
        if self._use_bias:
            ret += self._bias
        if self._activation is not None:
            ret = self._activation(ret)
        return ret
