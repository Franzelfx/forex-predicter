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
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization()

        self.feed_forward = self.build_feed_forward(features)
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization()

    def build_feed_forward(self, features):
        return tf.keras.Sequential(
            [
                Dense(self.hidden_neurons, activation="relu"),
                Dense(features),
            ]
        )

    def call(self, inputs, training=False):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.norm1(inputs + attention_output)

        feed_forward_output = self.feed_forward
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        block_output = self.norm2(attention_output + feed_forward_output)

        return block_output
