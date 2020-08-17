import tensorflow as tf
from layers.mha import MultiHeadAttention
from layers.ffn import PointWiseFeedForwardNetwork
from layers.position_embedding import PositionEmbedding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = PositionEmbedding(use_dynamic_slicing=True,
                                              max_sequence_length=self.maximum_position_encoding,
                                              name="encoder/position_embedding")

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training=True):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, feature_size) => (batch_size, input_seq_len, d_model)
        position_embedding = self.pos_encoding(x)
        x += position_embedding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(dff, d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=True):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), None, True)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, maximum_position_encoding=5000)

    sample_encoder_output = sample_encoder(tf.random.uniform((32, 1000, 26)),
                                           mask=None, training=False)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
