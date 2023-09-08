import tensorflow as tf
from tensorflow.keras.layers import InputSpec
from typing import List
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons, attention_heads, dropout_rate, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.hidden_neurons = hidden_neurons
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        self.dense_1 = Dense(hidden_neurons, activation="relu")
        self.dense_2 = Dense(hidden_neurons, activation="relu")
        self.dense_3 = Dense(hidden_neurons, activation="relu")
        self.multihead_attention = MultiHeadAttention(attention_heads, hidden_neurons)
        self.dropout_attention = Dropout(dropout_rate)
        self.concat_attention = Concatenate()
        self.layer_norm_1 = LayerNormalization()
        
        # Feed forward layers
        self.dense_ffn_1 = Dense(hidden_neurons, activation="relu")

        self.dropout_ffn = Dropout(dropout_rate)
        self.concat_ffn = Concatenate()
        self.layer_norm_2 = LayerNormalization()

    def call(self, input_tensor):
        input_matched_1 = self.dense_1(input_tensor)
        query = self.dense_2(input_matched_1)
        value = self.dense_3(input_matched_1)

        attention_1 = self.multihead_attention(query, value)
        dropout_attention = self.dropout_attention(attention_1)
        residual_attention = self.concat_attention([input_tensor, dropout_attention])
        norm_attention = self.layer_norm_1(residual_attention)

        feed_forward_1 = self.dense_ffn_1(norm_attention)

        dropout_ffn = self.dropout_ffn(feed_forward_1)
        residual_ffn = self.concat_ffn([norm_attention, dropout_ffn])
        norm_ffn = self.layer_norm_2(residual_ffn)

        return norm_ffn

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_neurons': self.hidden_neurons,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class TransformerLSTMBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_transformer,
        neurons_lstm,
        neurons_dense,
        attention_heads,
        dropout_rate,
        **kwargs
    ):
        super(TransformerLSTMBlock, self).__init__(**kwargs)
        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        self.transformer_block = TransformerBlock(
            neurons_transformer, attention_heads, dropout_rate
        )
        self.lstm_layer = LSTM(neurons_lstm, return_sequences=True)
        self.lstm_match = Dense(neurons_lstm, activation="relu")
        self.concat = Concatenate()

    def call(self, input_tensor):
        transformer = self.transformer_block(input_tensor)
        lstm = self.lstm_layer(transformer)
        lstm_match = self.lstm_match(lstm)
        concat = self.concat([transformer, lstm_match])
        return concat

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'neurons_transformer': self.neurons_transformer,
            'neurons_lstm': self.neurons_lstm,
            'neurons_dense': self.neurons_dense,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class Branch(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_transformer: List[int],
        neurons_lstm: List[int],
        neurons_dense: List[int],
        attention_heads: List[int],
        dropout_rate: List[float],
        **kwargs
    ):
        super(Branch, self).__init__(**kwargs)

        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        # Build the layers within the branch
        self.transformer_layers = []
        for (
            neurons_transformer,
            neurons_lstm,
            neurons_dense,
            attention_heads,
            dropout_rate,
        ) in zip(
            self.neurons_transformer,
            self.neurons_lstm,
            self.neurons_dense,
            self.attention_heads,
            self.dropout_rate,
        ):
            transformer_block = TransformerLSTMBlock(
                neurons_transformer,
                neurons_lstm,
                neurons_dense,
                attention_heads,
                dropout_rate,
            )
            self.transformer_layers.append(transformer_block)

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A BranchLayer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a BranchLayer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        # Build the layers within the branch
        self.transformer_layers = []
        for (
            neurons_transformer,
            neurons_lstm,
            neurons_dense,
            attention_heads,
            dropout_rate,
        ) in zip(
            self.neurons_transformer,
            self.neurons_lstm,
            self.neurons_dense,
            self.attention_heads,
            self.dropout_rate,
        ):
            transformer_block = TransformerLSTMBlock(
                neurons_transformer,
                neurons_lstm,
                neurons_dense,
                attention_heads,
                dropout_rate,
            )
            self.transformer_layers.append(transformer_block)
        self.built = True

    def call(self, inputs):
        x = inputs
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'neurons_transformer': self.neurons_transformer,
            'neurons_lstm': self.neurons_lstm,
            'neurons_dense': self.neurons_dense,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class Output(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_dense: List[int],
        dropout_rate: List[float],
        output_neurons: int,
        **kwargs,
    ):
        super(Output, self).__init__(**kwargs)
        assert len(neurons_dense) == len(
            dropout_rate
        ), "Length of neurons_dense and dropout_rate should be equal"

        self.neurons_dense = neurons_dense
        self.dropout_rate = dropout_rate
        self.output_neurons = output_neurons
        self.dense_layers = []
        self.dropout_layers = []
        self.gap = GlobalAveragePooling1D()
        self.output_layer = Dense(output_neurons, activation="linear")

        for neurons, dropout in zip(self.neurons_dense, self.dropout_rate):
            dense_layer = Dense(neurons, activation="linear")
            dropout_layer = Dropout(dropout)
            self.dense_layers.append(dense_layer)
            self.dropout_layers.append(dropout_layer)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.gap(x)
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'neurons_dense': self.neurons_dense,
            'dropout_rate': self.dropout_rate,
            'output_neurons': self.output_neurons,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

