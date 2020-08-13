import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        