import tensorflow as tf
from layers import Encoder, Decoder
from utils import create_padding_mask, create_look_ahead_mask


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, pe_input, pe_target, rate=0.1,
                 training=True):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        self.training = training

    def call(self, x):
        inp, enc_mask, tar = x

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(enc_mask, tar)

        enc_output = self.encoder(inp, enc_padding_mask, self.training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, combined_mask, dec_padding_mask, self.training)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    def create_masks(self, inp, tar):
        # 编码器填充遮挡
        if inp is None or tar is None:
            return None, None, None
        enc_padding_mask = create_padding_mask(inp)

        # 在解码器的第二个注意力模块使用。
        # 该填充遮挡用于遮挡编码器的输出。
        dec_padding_mask = create_padding_mask(inp)

        # 在解码器的第一个注意力模块使用。
        # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 62, 26))
    temp_target = tf.random.uniform((64, 100))

    fn_out, _ = sample_transformer(temp_input, temp_target, enc_mask=None, training=False)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
