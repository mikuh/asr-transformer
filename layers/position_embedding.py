import tensorflow as tf
from utils import tf_utils


class PositionEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 initializer="glorot_uniform",
                 use_dynamic_slicing=False,
                 max_sequence_length=None,
                 **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(PositionEmbedding, self).__init__(**kwargs)
        if use_dynamic_slicing and max_sequence_length is None:
            raise ValueError(
                "If `use_dynamic_slicing` is True, `max_sequence_length` must be set."
            )
        self._max_sequence_length = max_sequence_length
        self._initializer = tf.keras.initializers.get(initializer)
        self._use_dynamic_slicing = use_dynamic_slicing

    def build(self, input_shape):
        """Implements build() for the layer."""
        if not isinstance(input_shape, list):
            dimension_list = input_shape.as_list()
        else:
            dimension_list = input_shape

        if len(dimension_list) != 3:
            raise ValueError("PositionEmbedding expects a 3-dimensional input tensor "
                             "of shape [batch, sequence, width]")
        seq_length = dimension_list[1]
        width = dimension_list[2]

        # If we are not using dynamic slicing, we must assume that the sequence
        # length is fixed and max_sequence_length should not be specified.
        if not self._use_dynamic_slicing:
            if seq_length is None:
                raise ValueError(
                    "PositionEmbedding must have `use_dynamic_slicing` set "
                    "to True (and max_sequence_length set) when the "
                    "sequence (1st) dimension of the input is None.")
            if self._max_sequence_length is not None:
                raise ValueError(
                    "When `use_dynamic_slicing` is False, max_sequence_length should "
                    "not be specified and we ought to use seq_length to get the "
                    "variable shape.")

        if self._max_sequence_length is not None:
            weight_sequence_length = self._max_sequence_length
        else:
            weight_sequence_length = seq_length

        self._position_embeddings = self.add_weight(
            "embeddings",
            shape=[weight_sequence_length, width],
            initializer=self._initializer)

        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        """Implements call() for the layer."""
        if self._use_dynamic_slicing:
            input_shape = tf_utils.get_shape_list(inputs, expected_rank=3)
            seq_length = input_shape[1]
            width = input_shape[2]

            position_embeddings = tf.expand_dims(
                tf.slice(self._position_embeddings, [0, 0], [seq_length, width]), axis=0)
        else:
            position_embeddings = tf.expand_dims(self._position_embeddings, axis=0)
        return position_embeddings
