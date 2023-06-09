"""This module contains the model class for the LSTM model."""
import os
import logging
import numpy as np
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from datetime import datetime as dt
from keras.models import load_model
from keras.models import Model as KerasModel
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import (
    LSTM,
    Input,
    Dense,
    Bidirectional,
    LayerNormalization,
    MultiHeadAttention,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Model:
    """
    Model for Time Series Prediction.

    :param str path: Path to the directory where the model will be saved.
    :param str name: Name of the model.
    :param np.ndarray x_train: The training input data.
    :param np.ndarray y_train: The training output data.
    """

    def __init__(
        self,
        path: str,
        name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ):
        # Check if name has ":" in it, if so get characters after it
        if ":" in name:
            name = name.split(":")[1]
        self._name = name
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
        self._path = path
        self._x_train = x_train
        self._y_train = y_train
        self._model = None

    @property
    def steps_ahead(self) -> int:
        """Return the number of steps ahead that the model is capable of predicting."""
        return self._y_train.shape[1]

    def _build(self, hidden_neurons: int, dropout_rate: float, attention_heads: int, batch_size: int):
        input_shape = (self._x_train.shape[1], self._x_train.shape[2])
        inputs = Input(batch_shape=(batch_size,) + input_shape)

        # LSTM layer
        lstm_1 = Bidirectional(LSTM(hidden_neurons, return_sequences=True, stateful=True, batch_input_shape=(batch_size,) + input_shape))(inputs)
        dropout_1 = tf.keras.layers.Dropout(dropout_rate)(lstm_1)
        # Separate query and value branches for Attention layer
        query = Dense(hidden_neurons)(dropout_1)
        value = Dense(hidden_neurons)(dropout_1)

        # Apply Attention layer
        attention = MultiHeadAttention(attention_heads, hidden_neurons)(query, value)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        attention = LayerNormalization()(attention)

        dropout_2 = tf.keras.layers.Dropout(dropout_rate)(attention)
        lstm_2 = Bidirectional(LSTM(hidden_neurons, return_sequences=False, stateful=True, batch_input_shape=(batch_size,) + input_shape))(dropout_2)
        dense_1 = Dense(hidden_neurons, activation="relu")(lstm_2)
        dropout_3 = tf.keras.layers.Dropout(dropout_rate)(dense_1)
        dense_2 = Dense(hidden_neurons, activation="relu")(dropout_3)
        dropout_4 = tf.keras.layers.Dropout(dropout_rate)(dense_2)
        dense_3 = Dense(hidden_neurons, activation="relu")(dropout_4)
        dense_4 = Dense(hidden_neurons, activation="relu")(dense_3)
        output = Dense(self._y_train.shape[1], activation="linear")(dense_4)

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
        learning_rate=0.0001,
        hidden_neurons=32,
        dropout_rate: float = 0.4,
        attention_heads: int = 4,
        loss_fct: str = "mae",
        batch_size: int = 32,
        strategy=None,
    ):
        """Compile the model."""
        optimizer = Adam(learning_rate=learning_rate)
        # Check if multiple GPUs are available
        if strategy is not None and hasattr(strategy, "scope"):
            with strategy.scope():
                model = self._build(hidden_neurons, dropout_rate, attention_heads, batch_size)
                model.compile(loss=loss_fct, optimizer=optimizer, metrics=["mape"])
        else:
            model = self._build(hidden_neurons, dropout_rate, attention_heads, batch_size)
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

    def _adjust_sequence_length(self, data, batch_size):
        # Check for dimensionality
        if len(data.shape) == 2:
            samples, timesteps = data.shape
        elif len(data.shape) == 3:
            samples, timesteps, features = data.shape
        else:
            raise ValueError("Data must be 2D or 3D")
        new_timesteps = (timesteps // batch_size) * batch_size
        num_drop = timesteps - new_timesteps
        if len(data.shape) == 2:
            adjusted_data = data[:, :new_timesteps]
            if num_drop > 0:
                # Drop extra timesteps from the end of each sequence
                adjusted_data = adjusted_data[:, :-num_drop]
        elif len(data.shape) == 3:
            adjusted_data = data[:, :new_timesteps, :]
            if num_drop > 0:
                # Drop extra timesteps from the end of each sequence
                adjusted_data = adjusted_data[:, :-num_drop, :]

        return adjusted_data

    def fit(
        self,
        epochs=100,
        batch_size=32,
        patience=40,
        x_val=None,
        y_val=None,
        validation_split=0.2,
    ) -> DataFrame:
        """Compile and fit the model.

        :param int hidden_neurons: The number of neurons in the hidden layers.
        :param float dropout: The dropout factor for the dropout layers.
        :param str activation: The activation function for the hidden layers.
        :param int epochs: The number of epochs for the training process.
        :param float learning_rate: The learning rate for the training process.
        :param int batch_size: The batch size for the training process.
        :param str loss: The loss function for the training process.
        :param bool branched_model: If True, the model will be a branched model.
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
        # Adjust sequence length
        self._x_train = self._adjust_sequence_length(self._x_train, batch_size)
        self._y_train = self._adjust_sequence_length(self._y_train, batch_size)
        
        # Reshape the data to have the batch size as the first dimension
        num_batches = self._x_train.shape[0] // batch_size
        self._x_train = self._x_train[:num_batches * batch_size]
        self._y_train = self._y_train[:num_batches * batch_size]
        
        self._x_train = self._x_train.reshape(num_batches, batch_size, self._x_train.shape[1], self._x_train.shape[2])
        self._y_train = self._y_train.reshape(num_batches, batch_size, self._y_train.shape[1])
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
        # Set the validation split
        if (x_val and y_val) is not None:
            validation_split = 0
        # Fit the model
        try:
            fit = self._model.fit(
                self._x_train,
                self._y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val) if (x_val and y_val) is not None else None,
                validation_split=validation_split,
                callbacks=[tensorboard, model_checkpoint, early_stopping],
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
            return fit
        except Exception as e:
            # Print exception with traceback
            print(e)
            logging.error(e)

    def predict(
        self,
        x_input: np.ndarray,
        steps=1,
        scaler: MinMaxScaler = None,
        from_saved_model=False,
    ) -> np.ndarray:
        """Predict the output for the given input.

        :param np.ndarray x_input: The input data for the prediction as numpy array with shape (samples, time_steps, features).
        :param int steps: The number of steps to predict (one step is all steps for one sample).

        :remarks:   • If from_saved_model is False, "compile_and_fit()" must be called first.
                    • The predicted values are scaled back to the original scale if a scaler is given.
        """
        path = f"{self._path}/checkpoints/{self._name}_train.h5"
        # Get the model
        if from_saved_model:
            prediction_model = load_model(path)
            print(f"Loaded model from: {path}")
        else:
            # Check if the model has been fitted
            if self._model is None:
                raise Exception(
                    "The model has not been fitted yet, plase call compile_and_fit() first."
                )
            prediction_model = self._model
        # Predict the output
        # y_pred = model.predict(x_input, steps).flatten()
        print(f"Predict the output for {self._name}.")
        y_pred = prediction_model.predict(x_input, steps=steps, batch_size=32).flatten()
        # Reduce to only the output length
        y_pred = y_pred[: self._y_train.shape[1]]
        if scaler is not None:
            y_pred = y_pred.reshape(-1, 1)
            y_pred = scaler.inverse_transform(y_pred)
            y_pred = y_pred.flatten()
            print("Scaled back the prediction to original scale.")
        return y_pred
