import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputSpec
from typing import List
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons, max_position=5000, **kwargs):
        """
        Positional Encoding layer injects information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        
        Args:
            hidden_neurons (int): Dimension of the hidden neurons.
            max_position (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.hidden_neurons = hidden_neurons
        self.max_position = max_position

    def build(self, input_shape):
        # Compute positional encoding using sine and cosine functions with NumPy
        position = np.arange(self.max_position)[:, np.newaxis]  # (max_position, 1)
        div_term = np.exp(np.arange(0, self.hidden_neurons, 2) *
                          -(np.log(10000.0) / self.hidden_neurons))  # (hidden_neurons//2,)
        pe_sin = np.sin(position * div_term)  # (max_position, hidden_neurons//2)
        pe_cos = np.cos(position * div_term)  # (max_position, hidden_neurons//2)
        pe = np.concatenate([pe_sin, pe_cos], axis=1)  # (max_position, hidden_neurons)
        pe = np.expand_dims(pe, axis=0)  # (1, max_position, hidden_neurons)
        
        # Ensure the positional encoding is of type float32
        pe = pe.astype(np.float32)
        shape_pe = pe.shape  # e.g., (1, max_position, hidden_neurons)
        
        # Register positional encodings as a non-trainable weight
        self.pe = self.add_weight(
            name="pe",
            shape=shape_pe,  # Ensure 'shape' is passed only once
            initializer=tf.keras.initializers.Constant(pe),
            trainable=False
        )
        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        """
        Adds positional encoding to the input tensor.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, hidden_neurons)
        
        Returns:
            tf.Tensor: Tensor with positional encodings added.
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'hidden_neurons': self.hidden_neurons,
            'max_position': self.max_position,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons, attention_heads, dropout_rate, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.hidden_neurons = hidden_neurons
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])

        # Ensure hidden_neurons is divisible by attention_heads
        if self.hidden_neurons % self.attention_heads != 0:
            raise ValueError(
                f"hidden_neurons ({self.hidden_neurons}) must be divisible by attention_heads ({self.attention_heads})."
            )

        # Multi-Head Attention
        self.multihead_attention = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.hidden_neurons // self.attention_heads,
            dropout=self.dropout_rate,
            name="multihead_attention"
        )
        self.dropout_attention = Dropout(self.dropout_rate, name="dropout_attention")
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6, name="layer_norm_1")

        # Feed Forward Network
        self.dense_ffn_1 = Dense(self.hidden_neurons * 4, activation="relu", name="ffn_dense_1")
        self.dense_ffn_2 = Dense(self.hidden_neurons, activation=None, name="ffn_dense_2")
        self.dropout_ffn = Dropout(self.dropout_rate, name="dropout_ffn")
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6, name="layer_norm_2")

        # Projection layer for residual connection if input and hidden dimensions differ
        # Ensure the projection aligns with the number of hidden neurons (192 in this case)
        if self.input_dim != self.hidden_neurons:
            self.residual_projection = Dense(
                self.hidden_neurons, activation=None, name="residual_projection"
            )
        else:
            self.residual_projection = None

        super(TransformerBlock, self).build(input_shape)

    def call(self, input_tensor, training=False):
        # Compute Multi-Head Attention
        attention_output = self.multihead_attention(
            query=input_tensor,
            value=input_tensor,
            key=input_tensor,
            training=training
        )
        attention_output = self.dropout_attention(attention_output, training=training)

        # Apply residual connection with projection if necessary
        if self.residual_projection:
            # Project input_tensor to match the hidden_neurons (192 in this case)
            input_tensor_proj = self.residual_projection(input_tensor)
        else:
            input_tensor_proj = input_tensor

        # Ensure dimensions match before adding tensors
        if input_tensor_proj.shape[-1] != attention_output.shape[-1]:
            raise ValueError(f"Dimension mismatch: input_tensor_proj shape {input_tensor_proj.shape[-1]} != attention_output shape {attention_output.shape[-1]}")
        
        # Apply residual connection
        attention_output = input_tensor_proj + attention_output

        # Layer normalization after residual connection
        attention_output = self.layer_norm_1(attention_output)

        # Feed Forward Network
        ffn_output = self.dense_ffn_1(attention_output)
        ffn_output = self.dense_ffn_2(ffn_output)
        ffn_output = self.dropout_ffn(ffn_output, training=training)

        # Residual connection (no projection needed here)
        ffn_output = attention_output + ffn_output

        # Layer normalization after FFN residual connection
        output = self.layer_norm_2(ffn_output)

        return output

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
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

    def build(self, input_shape):
        # Initialize Transformer Block
        self.transformer_block = TransformerBlock(
            hidden_neurons=self.neurons_transformer,
            attention_heads=self.attention_heads,
            dropout_rate=self.dropout_rate,
            name=f"transformer_block_{self.neurons_transformer}"
        )

        # Initialize LSTM layer if specified
        if self.neurons_lstm > 0:
            self.lstm_layer = LSTM(
                self.neurons_lstm,
                return_sequences=True,
                name=f"lstm_layer_{self.neurons_lstm}"
            )
            # Adjust Dense layer to match Transformer output dimensions for addition
            self.lstm_match = Dense(
                self.neurons_transformer,
                activation="relu",
                name=f"lstm_match_{self.neurons_transformer}"
            )
        else:
            self.lstm_layer = None

        # Projection layer for residual connection if necessary
        if self.lstm_layer is not None and self.neurons_lstm != self.neurons_transformer:
            self.residual_projection = Dense(
                self.neurons_transformer,
                activation=None,
                name=f"residual_projection_{self.neurons_transformer}"
            )
        else:
            self.residual_projection = None

        super(TransformerLSTMBlock, self).build(input_shape)

    def call(self, input_tensor, training=False):
        # Pass through Transformer Block
        transformer_output = self.transformer_block(input_tensor, training=training)

        if self.lstm_layer is not None:
            # Pass through LSTM
            lstm_output = self.lstm_layer(transformer_output, training=training)
            # Match LSTM output dimensions to Transformer for residual addition
            lstm_matched = self.lstm_match(lstm_output, training=training)

            # Apply residual projection if necessary
            if self.residual_projection:
                transformer_output_proj = self.residual_projection(transformer_output)
                output = transformer_output_proj + lstm_matched
            else:
                output = transformer_output + lstm_matched
        else:
            output = transformer_output

        return output

    def get_config(self):
        config = super(TransformerLSTMBlock, self).get_config()
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
        name=None, 
        **kwargs
    ):
        super(Branch, self).__init__(name=name, **kwargs)
        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

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
        self.input_spec = InputSpec(min_ndim=3, axes={-1: last_dim})

        # Initialize Input Projection to match the first hidden_neurons in transformer layers
        if len(self.neurons_transformer) == 0:
            raise ValueError("neurons_transformer list must contain at least one element.")
        self.input_projection = Dense(
            self.neurons_transformer[0],
            activation=None,
            name="input_projection"
        )

        # Initialize Positional Encoding
        self.positional_encoding = PositionalEncoding(
            hidden_neurons=self.neurons_transformer[0],
            name="positional_encoding"
        )

        # Build the Transformer-LSTM layers with projections for differing sizes
        self.transformer_layers = []
        for idx, (
            neurons_transformer,
            neurons_lstm,
            neurons_dense,
            attention_head,
            dropout
        ) in enumerate(zip(
            self.neurons_transformer,
            self.neurons_lstm,
            self.neurons_dense,
            self.attention_heads,
            self.dropout_rate,
        )):
            transformer_block = TransformerLSTMBlock(
                neurons_transformer=neurons_transformer,
                neurons_lstm=neurons_lstm,
                neurons_dense=neurons_dense,
                attention_heads=attention_head,
                dropout_rate=dropout,
                name=f"transformer_lstm_block_{idx}"
            )
            self.transformer_layers.append(transformer_block)

            # Add a projection layer if next transformer layer has a different size
            if idx < len(self.neurons_transformer) - 1:
                next_transformer_size = self.neurons_transformer[idx + 1]
                if neurons_transformer != next_transformer_size:
                    projection_layer = Dense(
                        next_transformer_size,
                        activation=None,
                        name=f"projection_to_{next_transformer_size}_{idx}"
                    )
                    self.transformer_layers.append(projection_layer)

        self.built = True

    def call(self, inputs, training=False):
        # Project input to the first hidden_neurons in transformer
        x = self.input_projection(inputs, training=training)
        # Apply Positional Encoding
        x = self.positional_encoding(x)
        # Pass through Transformer-LSTM blocks and apply projections if necessary
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)
        return x

    def get_config(self):
        config = super(Branch, self).get_config().copy()
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
        """
        Output layer consisting of Dense and Dropout layers followed by a final output layer.
        
        Args:
            neurons_dense (List[int]): List of neuron counts for Dense layers.
            dropout_rate (List[float]): List of dropout rates.
            output_neurons (int): Number of neurons in the final output layer.
        """
        super(Output, self).__init__(**kwargs)
        assert len(neurons_dense) == len(
            dropout_rate
        ), "Length of neurons_dense and dropout_rate should be equal"

        self.neurons_dense = neurons_dense
        self.dropout_rate = dropout_rate
        self.output_neurons = output_neurons

    def build(self, input_shape):
        self.dense_layers = [
            Dense(neurons, activation="relu", name=f"dense_{i}")
            for i, neurons in enumerate(self.neurons_dense)
        ]
        self.dropout_layers = [
            Dropout(rate, name=f"dropout_{i}")
            for i, rate in enumerate(self.dropout_rate)
        ]
        
        self.gap = GlobalAveragePooling1D(name="global_average_pooling")
        self.output_layer = Dense(
            self.output_neurons, activation="linear", name="output_layer"
        )

        super(Output, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.gap(inputs)
        
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x)
            
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super(Output, self).get_config()
        config.update({
            'neurons_dense': self.neurons_dense,
            'dropout_rate': self.dropout_rate,
            'output_neurons': self.output_neurons,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
