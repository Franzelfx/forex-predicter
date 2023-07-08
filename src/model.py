"""This module contains the model class for the LSTM model."""
import os
import logging
import numpy as np
from typing import Any, List
import tensorflow as tf
from pandas import DataFrame
from tensorflow import Tensor
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from datetime import datetime as dt
from keras.models import load_model
from keras.models import Model as KerasModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau,
)
from keras.layers import (
    Add,
    LSTM,
    Input,
    Dense,
    Dropout,
    Bidirectional,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Branch:
    def __init__(
        self,
        _input: np.ndarray,
        hidden_neurons,
        lstm_neurons,
        dense_neurons,
        attention_heads,
        dropout_rate,
    ):
        self.input = _input
        self.transformer_neurons = hidden_neurons
        self.lstm_neurons = lstm_neurons
        self.dense_neurons = dense_neurons
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate


class Output:
    def __init__(self, hidden_neurons, dropout_rate):
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate


class Architecture:
    def __init__(self, branches: List[Branch], main_branch: Branch, output: Output):
        self.branches: List[Branch] = branches
        self.main_branch: Branch = main_branch
        self.output: Output = output


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()


class TransformerBlock:
    def __init__(self, hidden_neurons, attention_heads, dropout_rate):
        self.hidden_neurons = hidden_neurons
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

    def __call__(self, input_tensor):
        input_matched_1 = Dense(self.hidden_neurons)(input_tensor)

        # Separate query and value branches for Attention layer
        query = Dense(self.hidden_neurons)(input_matched_1)
        value = Dense(self.hidden_neurons)(input_matched_1)

        # Apply Attention layer
        attention_1 = MultiHeadAttention(self.attention_heads, self.hidden_neurons)(
            query, value
        )

        # Add dropout and residual connection and layer normalization
        dropout_attention = Dropout(self.dropout_rate)(attention_1)
        residual_attention = Add()([input_matched_1, dropout_attention])
        norm_attention = LayerNormalization()(residual_attention)

        # Feed forward layer
        feed_forward_1 = Dense(self.hidden_neurons, activation="relu")(norm_attention)
        feed_forward_2 = Dense(self.hidden_neurons, activation="relu")(feed_forward_1)
        feed_forward_3 = Dense(self.hidden_neurons, activation="relu")(feed_forward_2)

        # Add dropout, residual connection, and layer normalization
        dropout_ffn = Dropout(self.dropout_rate)(feed_forward_3)
        residual_ffn = Add()([norm_attention, dropout_ffn])
        norm_ffn = LayerNormalization()(residual_ffn)

        return norm_ffn


class TransformerLSTMBlock:
    def __init__(
        self,
        neurons_transformer,
        neurons_lstm,
        neurons_dense,
        attention_heads,
        dropout_rate,
    ):
        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.transformer_block = TransformerBlock(
            neurons_transformer, attention_heads, dropout_rate
        )

    def __call__(self, input_tensor):
        transformer_block = self.transformer_block(input_tensor)
        lstm = Bidirectional(LSTM(self.neurons_lstm, return_sequences=True))(
            input_tensor
        )
        lstm_matched = Dense(self.neurons_dense)(lstm)
        added = Add()([transformer_block, lstm_matched])
        norm = LayerNormalization()(added)
        return norm


class BranchLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        neurons_transformer: List[int],
        neurons_lstm: List[int],
        neurons_dense: List[int],
        attention_heads: List[int],
        dropout_rate: List[float],
        output_neurons: int,
        **kwargs,
    ):
        super(BranchLayer, self).__init__(**kwargs)
        self.neurons_transformer = neurons_transformer
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.output_neurons = output_neurons
        self.model = None

    def call(self, inputs, **kwargs):
        x = inputs
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
            block = TransformerLSTMBlock(
                neurons_transformer,
                neurons_lstm,
                neurons_dense,
                attention_heads,
                dropout_rate,
            )
            x = block(x)
        x = GlobalAveragePooling1D()(x)

        i = 0
        for neurons in self.neurons_dense:
            x = Dense(neurons, activation="relu")(x)
            x = Dropout(self.dropout_rate[i])(x)
            i += 1
        output = Dense(self.output_neurons, activation="linear")(x)
        return output


class OutputLayer(tf.keras.layers.Layer):
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

    def call(self, inputs, **kwargs):
        x = inputs
        for neurons, dropout in zip(self.neurons_dense, self.dropout_rate):
            x = Dense(neurons, activation="relu")(x)
            x = Dropout(dropout)(x)

        # Final output layer
        output = Dense(self.output_neurons, activation="linear")(x)
        return output


class Model:
    """
    Model for Time Series Prediction
    =====

    Parameters
    ----------
    path
        :obj:`str` -> Path to save the model
    name
        :obj:`str` -> Name of the model
    x_train
        :obj:`np.ndarray` -> Training data input shape (samples, time_steps_in, features)
    y_train
        :obj:`np.ndarray` -> Training data output shape (samples, time_steps_out)
    batch_size
        :obj:`int` -> Batch size for training in fit method

    Attributes
    ----------
    steps_ahead
        :obj:`int`
        Number of steps ahead to predict

    Examples
    --------
    >>> from src.model import Model
    >>> model = Model(
    ...     path="models",
    ...     name="model",
    ...     x_train=x_train,
    ...     y_train=y_train,
    ...     batch_size=1,
    ... )
    >>> model.compile()
    >>> model.fit(epochs=10)
    >>> model.predict(x_test)
    ...
    """

    def __init__(
        self,
        path: str,
        name: str,
        y_train: np.ndarray,
    ):
        # Check if name has ":" in it, if so get characters after it
        if ":" in name:
            name = name.split(":")[1]
        self._name = name
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
        self._path = path
        self._y_train = y_train
        # Model structure variables
        self._branches = []
        self._summation_neurons_dense = []
        self._summation_dropout_rate = []
        # Main branch variables
        self._main_branch = None
        self._main_branch_neurons_transformer = []
        self._main_branch_neurons_lstm = []
        self._main_branch_neurons_dense = []
        self._main_branch_attention_heads = []
        self._main_branch_dropout_rate = []
        # Output variables
        self._output: Output = None
        self._model = None

    @property
    def steps_ahead(self) -> int:
        """Get the steps ahead."""
        return self._y_train.shape[1]

    def _var_name(self, var):
        variable_names = [
            tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())
        ]
        if variable_names:
            return variable_names[0]
        else:
            return None

    def _inverse_transform(
        self, scaler: StandardScaler, data: np.ndarray
    ) -> np.ndarray:
        """Inverse transform the data."""
        data = data.reshape(-1, 1)
        data = scaler.inverse_transform(data).flatten()
        return data

    def _build(self, architecture: Architecture) -> KerasModel:
        """Build the model."""
        # Main branch
        inputs = []
        branches = []
        for branch in architecture.branches:
            input_shape = (branch.input.shape[1], branch.input.shape[2])
            _input = Input(shape=input_shape)
            _branch = BranchLayer(
                branch.transformer_neurons,
                branch.lstm_neurons,
                branch.dense_neurons,
                branch.attention_heads,
                branch.dropout_rate,
                self._y_train.shape[1],
            )(_input)
            branches.append(_branch)
            inputs.append(_input)
        # Summation layer
        summation = Add()(branches)
        # Main branch
        main_branch = BranchLayer(
            architecture.main_branch.transformer_neurons,
            architecture.main_branch.lstm_neurons,
            architecture.main_branch.dense_neurons,
            architecture.main_branch.attention_heads,
            architecture.main_branch.dropout_rate,
            self._y_train.shape[1],
        )(summation)
        # Output layer
        output = OutputLayer(
            architecture.output_neurons,
            architecture.dropout_rate,
            architecture.output_neurons,
        )(main_branch)
        # Build the model
        model = KerasModel(inputs=inputs, outputs=output)
        return model

    def _plot_fit_history(self, fit):
        """Plot the fit history."""
        # High resolution plot with subplots
        date = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.cla()
        plt.clf()
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        # High resolution plot
        fig.set_dpi(300)
        # Plot the loss
        axes[0].plot(fit.history["loss"], label="loss")
        axes[0].plot(fit.history["val_loss"], label="val_loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid()
        # Plot the metrics
        axes[1].plot(fit.history["mape"], label="mape")
        axes[1].plot(fit.history["val_mape"], label="val_mape")
        axes[1].set_ylabel("MAPE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("MAPE")
        axes[1].legend()
        axes[1].grid()
        # Tight layout
        fig.tight_layout()
        # Set plot title
        plt.suptitle(f"Fit History {self._name} {date}")
        # Save the plot
        plt.savefig(f"{self._path}/fit_history/{self._name}.png")

    def compile(
        self,
        architecture: Architecture,
        learning_rate=0.0001,
        loss_fct: str = "mae",
        strategy=None,
    ):
        """Compile the model."""
        optimizer = Adam(learning_rate=learning_rate)
        # Check if multiple GPUs are available
        if strategy is not None and hasattr(strategy, "scope"):
            with strategy.scope():
                model = self._build(architecture)
                model.compile(loss=loss_fct, optimizer=optimizer, metrics=["mape"])
        else:
            model = self._build(architecture)
            model.compile(loss=loss_fct, optimizer=optimizer, metrics=["mape"])
        model.summary()
        # Plot the model
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=f"{self._path}/models/{self._name}.png",
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=False,
                dpi=300,
            )
        except Exception as e:
            # Print exception with traceback
            print(e)
            logging.error(e)
        self._model = model

    def fit(
        self,
        epochs=100,
        batch_size=32,
        patience=40,
        validation_split=0.1,
    ) -> DataFrame:
        """Compile and fit the model.

        :param int hidden_neurons: The number of neurons in the hidden layers.
        :param float dropout: The dropout factor for the dropout layers.
        :param str activation: The activation function for the hidden layers.
        :param int epochs: The number of epochs for the training process.
        :param float learning_rate: The learning rate for the training process.
        :param int batch_size: The batch size for the training process.
        :param str loss: The loss function for the training process.
        :param int patience: The patience for the early stopping callback.
        :param np.ndarray x_val: The validation input data.
        :param np.ndarray y_val: The validation output data.
        :param float validation_split: The validation split for the training process.

        :returns: The fit history.

        :remarks:   • The metric for this model is fix and is the mean absolute percentage error (MAPE).
                    • If no validation data is given, the validation split will be used.
                    • The model is saved in the checkpoints folder.
                    • The validation loss is saved in the fit_history folder.
                    • The tensorboard logs are saved in the tensorboard folder.
        """
        # Say how much GPU's are available
        if self._model is None:
            print("Model is not compiled yet, please compile the model first.")
            return
        reset_states = ResetStatesCallback()
        model_checkpoint = ModelCheckpoint(
            filepath=f"{self._path}/checkpoints/{self._name}_train.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", verbose=1
        )
        tensorboard = TensorBoard(log_dir=f"{self._path}/tensorboard/{self._name}")
        lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=30, min_lr=0.000001)
        # Split the data
        X_train = []
        X_val = []
        for branch in self._branches:
            _X_train, _X_val = train_test_split(
                branch.X_train, test_size=validation_split, shuffle=False
            )
            X_train.append(_X_train)
            X_val.append(_X_val)
        y_train, y_val = train_test_split(
            self._y_train, test_size=validation_split, shuffle=False
        )
        # Fit the model
        try:
            fit = self._model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                validation_split=validation_split,
                callbacks=[
                    tensorboard,
                    model_checkpoint,
                    early_stopping,
                    reset_states,
                    lr_scheduler,
                ],
                shuffle=False,
            )
            # Load the best weights
            self._model.load_weights(f"{self._path}/checkpoints/{self._name}_train.h5")
            self._model = self._model
            self._plot_fit_history(fit)
            # Convert the fit history to dataframe
            fit = DataFrame(fit.history)
            # Save the fit history
            fit.to_csv(f"{self._path}/fit_history/{self._name}.csv", index=False)
        except Exception as e:
            # Print exception with traceback
            print(e)
            logging.error(e)
        return fit

    def predict(
        self,
        x_hat: np.ndarray,
        x_train: np.ndarray = np.array([]),
        x_test: np.ndarray = np.array([]),
        scaler: StandardScaler = None,
        from_saved_model=False,
    ) -> np.ndarray:
        """Predict the output for the given input.
        :param np.ndarray x_input: The input data for the prediction as numpy array with shape (samples, time_steps, features).
        :param int steps: The number of steps to predict (one step is all steps for one sample).

        :remarks:   • If from_saved_model is False, "compile_and_fit()" must be called first.
                    • The predicted values are scaled back to the original scale if a scaler is given.
        """
        y_train = None
        y_test = None
        path = f"{self._path}/checkpoints/{self._name}_train.h5"
        # Get the model
        if from_saved_model:
            prediction_model: tf.keras.Model = load_model(path)
            print(f"Loaded model from: {path}")
        else:
            # Check if the model has been fitted
            if self._model is None:
                raise Exception(
                    "The model has not been fitted yet, plase call compile_and_fit() first."
                )
            prediction_model = self._model
        # Prepare the input
        print(f"x_train samples: {x_train.shape[0]}")
        print(f"x_test samples: {x_test.shape[0]}")
        print(f"x_hat samples: {x_hat.shape[0]}")
        # BUG: wrong vstacking
        if x_train.shape[0] > 0:
            x = np.vstack((x_train, x_test))
            x = np.vstack((x, x_hat))
        else:
            if x_test.shape[0] > 0:
                x = np.vstack((x_test, x_hat))
            else:
                x = x_hat
        # Predict the output
        print(f"Predicting {x.shape[0]} samples with {x.shape[1]} timesteps.")
        y_hat = []
        for i in range(0, len(x), self._batch_size):
            x_batch = x[i : i + self._batch_size]
            y_batch = prediction_model.predict(
                x_batch, batch_size=self._batch_size, verbose=0
            ).flatten()
            y_hat.append(y_batch)
        # Extract the predicted values
        # When x_train was given, we need to extract x_train from the prediction,
        # this will be done by taking the last x_train.shape[0] samples from the prediction.
        if x_train is not None:
            y_train = y_hat[-x_train.shape[0] :]
            # Drop the extracted from the list
            y_hat = y_hat[: -x_train.shape[0]]
            y_train = np.array(y_train).flatten()
        # Extract also the test data if given and drop it from the list.
        if x_test is not None:
            y_test = y_hat[-x_test.shape[0] :]
            y_hat = y_hat[: -x_test.shape[0]]
            y_test = np.array(y_test).flatten()
        # Flatten the list
        y_hat = np.array(y_hat).flatten()

        # Scale the output back to the original scale
        if scaler is not None:
            y_hat = self._inverse_transform(scaler, y_hat)
            if x_train is not None:
                y_train = self._inverse_transform(scaler, y_train)
            if x_test is not None:
                y_test = self._inverse_transform(scaler, y_test)

        # Return the predicted values, based on the given input
        return y_train, y_test, y_hat
