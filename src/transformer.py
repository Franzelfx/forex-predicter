import tensorflow as tf
from keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Add

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, features, hidden_neurons, num_attention_heads, key_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.num_attention_heads = num_attention_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

        self.attention = MultiHeadAttention(
            num_attention_heads, key_dim=key_dim
        )
        self.dropout_1 = Dropout(dropout_rate)
        self.norm_1 = LayerNormalization()

        self.feed_forward_1 = Dense(hidden_neurons, activation="relu")
        self.feed_forward_2 = Dense(features)
        self.dropout_2 = Dropout(dropout_rate)
        self.norm_2 = LayerNormalization()


    def call(self, inputs, training=False):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout_1(attention_output, training=training)
        attention_output = self.norm_1(inputs + attention_output)

        feed_forward_output = self.feed_forward_1(attention_output)
        feed_forward_output = self.feed_forward_2(feed_forward_output)
        feed_forward_output = self.dropout_2(feed_forward_output, training=training)
        block_output = self.norm_2(attention_output + feed_forward_output)

        return block_output
