import tensorflow as tf

_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]


class DenseEinsum(tf.keras.layers.Layer):

    def __init__(self,
                 output_shape,
                 num_summed_dimensions=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseEinsum, self).__init__(**kwargs)
        self._output_shape = output_shape if isinstance(output_shape, (list, tuple)) else (output_shape,)
        self._activation = tf.keras.activations.get(activation)
        pass

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
        pass


