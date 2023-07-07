import tensorflow as tf
from keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Add


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons, attention_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        self.query_branch = Dense(hidden_neurons)
        self.value_branch = Dense(hidden_neurons)

        self.attention = MultiHeadAttention(attention_heads, hidden_neurons)
        self.dropout_attention = Dropout(dropout_rate)
        self.norm_attention = LayerNormalization()

        self.feed_forward_1 = Dense(hidden_neurons, activation="relu")
        self.feed_forward_2 = Dense(hidden_neurons, activation="relu")
        self.feed_forward_3 = Dense(hidden_neurons, activation="relu")

        self.dropout_ffn = Dropout(dropout_rate)
        self.norm_ffn = LayerNormalization()

    def call(self, inputs, training=False):
        query = self.query_branch(inputs)
        value = self.value_branch(inputs)

        attention_output = self.attention(query, value)
        dropout_attention = self.dropout_attention(attention_output, training=training)
        residual_attention = Add()([inputs, dropout_attention])
        norm_attention = self.norm_attention(residual_attention)

        feed_forward_output = self.feed_forward_1(norm_attention)
        feed_forward_output = self.feed_forward_2(feed_forward_output)
        feed_forward_output = self.feed_forward_3(feed_forward_output)

        dropout_ffn = self.dropout_ffn(feed_forward_output, training=training)
        residual_ffn = Add()([norm_attention, dropout_ffn])
        norm_ffn = self.norm_ffn(residual_ffn)

        return norm_ffn
