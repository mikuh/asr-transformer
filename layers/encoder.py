import tensorflow as tf
from layers.attention import MultiHeadAttention
from layers.dense_einsum import DenseEinsum
from layers.position_embedding import PositionEmbedding
from utils.activations import gelu


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 num_layers,
                 intermediate_size,
                 initializer="glorot_uniform",
                 max_sequence_length=5120,
                 dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 is_training=True,
                 return_all_encoder_outputs=False,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self._return_all_encoder_outputs = return_all_encoder_outputs
        self._linear_dense = tf.keras.layers.Dense(d_model, activation=None)
        self._norm = tf.keras.layers.LayerNormalization(
            name="encoder/input_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)

        self._position_embedding = PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_sequence_length=max_sequence_length,
            name='encoder/position_embeddings')

        self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # encoder layers
        self._encoder_layers = []
        for i in range(num_layers):
            self._encoder_layers.append(EncoderLayer(d_model=d_model,
                                                     intermediate_size=intermediate_size,
                                                     num_heads=num_heads,
                                                     dropout_rate=dropout_rate,
                                                     attention_dropout_rate=attention_dropout_rate,
                                                     kernel_initializer=kernel_initializer,
                                                     bias_initializer=bias_initializer,
                                                     kernel_regularizer=kernel_regularizer,
                                                     bias_regularizer=bias_regularizer,
                                                     kernel_constraint=kernel_constraint,
                                                     bias_constraint=bias_constraint,
                                                     is_training=is_training,
                                                     ))

    def call(self, inputs):
        features, mask = inputs

        voice_features = self._linear_dense(features)
        voice_features = self._norm(voice_features)
        voice_position = self._position_embedding(voice_features)

        voice_embeddings = self._dropout(voice_features + voice_position)

        data = voice_embeddings
        encoder_outputs = []
        for layer in self._encoder_layers:
            data = layer([data, mask])
            encoder_outputs.append(data)

        first_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(encoder_outputs[-1])
        cls_output = self._cls_output_layer(first_token_tensor)

        if self._return_all_encoder_outputs:
            outputs = [encoder_outputs, cls_output]
        else:
            outputs = [encoder_outputs[-1], cls_output]
        return outputs


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
                                       dropout_rate=attention_dropout_rate,
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
            name="encoder/self_attention_output")

        self._attention_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        # Use float32 in layernorm for numeric stability.
        # It is probably safe in mixed_float16, but we haven't validated this yet.
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="encoder/self_attention_layer_norm",
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
