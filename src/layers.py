import tensorflow.compat.v2 as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.input_spec import InputSpec

from typing import List

from keras.layers import (
    Add,
    LSTM,
    Input,
    Dense,
    Dropout,
    Softmax,
    Bidirectional,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons, attention_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.dense_1 = Dense(hidden_neurons)
        self.dense_2 = Dense(hidden_neurons)
        self.dense_3 = Dense(hidden_neurons)
        self.multihead_attention = MultiHeadAttention(attention_heads, hidden_neurons)
        self.dropout_attention = Dropout(dropout_rate)
        self.dropout_ffn = Dropout(dropout_rate)
        self.add_1 = Add()
        self.add_2 = Add()
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()

    def call(self, input_tensor):
        input_matched_1 = self.dense_1(input_tensor)
        query = self.dense_2(input_matched_1)
        value = self.dense_3(input_matched_1)

        attention_1 = self.multihead_attention(query, value)
        dropout_attention = self.dropout_attention(attention_1)
        residual_attention = self.add_1([input_matched_1, dropout_attention])
        norm_attention = self.layer_norm_1(residual_attention)

        feed_forward_1 = Dense(self.hidden_neurons, activation="relu")(norm_attention)
        feed_forward_2 = Dense(self.hidden_neurons, activation="relu")(feed_forward_1)
        feed_forward_3 = Dense(self.hidden_neurons, activation="relu")(feed_forward_2)

        dropout_ffn = self.dropout_ffn(feed_forward_3)
        residual_ffn = self.add_2([norm_attention, dropout_ffn])
        norm_ffn = self.layer_norm_2(residual_ffn)

        return norm_ffn


class TransformerLSTMBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_transformer,
        neurons_lstm,
        neurons_dense,
        attention_heads,
        dropout_rate,
    ):
        super(TransformerLSTMBlock, self).__init__()
        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.transformer_block = TransformerBlock(
            neurons_transformer, attention_heads, dropout_rate
        )
        self.lstm_layer = Bidirectional(LSTM(neurons_lstm, return_sequences=True))
        self.dense_lstm = Dense(neurons_dense)
        self.add = Add()
        self.layer_norm = LayerNormalization()

    def call(self, input_tensor):
        transformer_block = self.transformer_block(input_tensor)
        lstm = self.lstm_layer(input_tensor)
        lstm_matched = self.dense_lstm(lstm)
        added = self.add([transformer_block, lstm_matched])
        norm = self.layer_norm(added)
        return norm


class Branch(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_transformer: List[int],
        neurons_lstm: List[int],
        neurons_dense: List[int],
        attention_heads: List[int],
        dropout_rate: List[float],
        output_neurons: int,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.output_neurons = int(output_neurons) if not isinstance(output_neurons, int) else output_neurons
        if self.output_neurons < 0:
            raise ValueError(
                "Received an invalid value for `output_neurons`, expected "
                f"a positive integer. Received: output_neurons={output_neurons}"
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
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
        
        self.dense_layers = []
        for neurons, dropout in zip(self.neurons_dense, self.dropout_rate):
            dense_layer = Dense(neurons, activation="relu")
            dropout_layer = Dropout(dropout)
            self.dense_layers.append((dense_layer, dropout_layer))
        
        self.output_layer = Dense(self.output_neurons, activation=self.activation)
        
        self.built = True

    def call(self, inputs):
        x = inputs
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        x = GlobalAveragePooling1D()(x)
        
        for dense_layer, dropout_layer in self.dense_layers:
            x = dense_layer(x)
            x = dropout_layer(x)
        
        outputs = self.output_layer(x)
        
        return outputs

class OutputLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_dense: List[int],
        dropout_rate: List[float],
        output_neurons: int,
        **kwargs,
    ):
        super(OutputLayer, self).__init__(**kwargs)
        assert len(neurons_dense) == len(
            dropout_rate
        ), "Length of neurons_dense and dropout_rate should be equal"

        self.neurons_dense = neurons_dense
        self.dropout_rate = dropout_rate
        self.output_neurons = output_neurons
        self.dense_layers = []
        self.dropout_layers = []
        self.output_layer = Dense(output_neurons)
        self.softmax_layer = Softmax(axis=-1)

        for neurons, dropout in zip(self.neurons_dense, self.dropout_rate):
            dense_layer = Dense(neurons, activation="relu")
            dropout_layer = Dropout(dropout)
            self.dense_layers.append(dense_layer)
            self.dropout_layers.append(dropout_layer)

    def call(self, inputs, **kwargs):
        x = inputs
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)
        output = self.softmax_layer(x)
        return output
