import tensorflow as tf
from layers.attention import MultiHeadAttention
from layers.ffn import PointWiseFFN


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 d_model,
                 intermediate_size,
                 num_heads,
                 dropout_rate,
                 attention_dropout_rate,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 is_training=True,
                 **kwargs):
        super(DecoderLayer, self).__init__()

        self.head_size = d_model // num_heads

        self._mha1 = MultiHeadAttention(num_heads=num_heads,
                                        head_size=self.head_size,
                                        dropout_rate=attention_dropout_rate,
                                        is_training=is_training,
                                        **kwargs)

        self._mha2 = MultiHeadAttention(num_heads=num_heads,
                                        head_size=self.head_size,
                                        dropout_rate=attention_dropout_rate,
                                        is_training=is_training,
                                        **kwargs)

        self._ffn = PointWiseFFN(d_model=d_model,
                                 intermediate_size=intermediate_size,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 kernel_constraint=kernel_constraint,
                                 bias_initializer=bias_initializer,
                                 bias_regularizer=bias_regularizer,
                                 bias_constraint=bias_constraint)

        self._layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self._dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self._dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self._dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
