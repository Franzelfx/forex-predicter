"""This module contains the model class for the LSTM model."""
import os
import logging
import traceback
import numpy as np
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model as KerasModel
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Flatten,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    concatenate,
    Bidirectional,
    TimeDistributed,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Model:
    """Used to create, compile, fit and predict with the LSTM model."""

    def __init__(
        self,
        path: str,
        name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """Set the fundamental attributes.

        @param path: The top level path to the checkpoint, fit, model and tensorboard folder.
        @param name: The name of the model (e.g. EURUSD in case of finacial analysis).
        @param x_train: The input data for the model.
        @param y_train: The output data for the model.
        """
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64' 
        self._path = path
        self._name = name
        self._x_train = x_train
        self._y_train = y_train
        self._model = None

    @property
    def steps_ahead(self) -> int:
        """Return the number of steps ahead that the model is capable of predicting."""
        return self._y_train.shape[1]

    def _adjusted_batch_size(self, desired_batch_size: int) -> int:
        """Return the adjusted batch size.

        The batch size must be divisible by the number of samples in the training set.
        """
        return desired_batch_size - (self._x_train.shape[0] % desired_batch_size)

    def _create_model(
        self, hidden_neurons: int, dropout_factor: float, activation: str, stateful: bool = False, batch_size: int = 32
    ) -> Sequential:
        model = Sequential()
        if(stateful):
            model.add(
                Bidirectional(
                    LSTM(
                        hidden_neurons,
                        return_sequences=True,
                        input_shape=(batch_size, self._x_train.shape[1], self._x_train.shape[2]),
                        stateful=stateful,
                    )
                )
            )
            model.add(
                Bidirectional(
                    LSTM(
                        hidden_neurons,
                        return_sequences=True,
                        batch_input_shape=(self._x_train.shape[1], self._x_train.shape[2]),
                        stateful=stateful,
                    )
                )
            )
        model.add(Bidirectional(LSTM(hidden_neurons, return_sequences=True, stateful=stateful)))
        model.add(Bidirectional(LSTM(round(0.5 * hidden_neurons), return_sequences=True, stateful=stateful)))
        model.add(TimeDistributed(Dense(round(0.75 * hidden_neurons), activation=activation)))
        model.add(Dropout(dropout_factor))
        model.add(TimeDistributed(Dense(round(0.75 * hidden_neurons), activation=activation)))
        model.add(Dropout(dropout_factor))
        model.add(TimeDistributed(Dense(round(0.5 * hidden_neurons), activation=activation)))
        model.add(TimeDistributed(Dense(self._y_train.shape[1], activation=activation)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(round(0.5 * hidden_neurons), activation=activation))
        model.add(Dense(round(0.5 * hidden_neurons), activation=activation))
        model.add(Dense(self._y_train.shape[1], activation="linear"))
        if(stateful):
            model.build(input_shape=(batch_size, self._x_train.shape[1], self._x_train.shape[2]))
        else:
            model.build(input_shape=(self._x_train.shape[0], self._x_train.shape[1], self._x_train.shape[2]))
        return model


    def _plot_fit_history(self, fit):
        """Plot the fit history."""
        # High resolution plot with subplots
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
        # Save the plot
        plt.savefig(f"{self._path}/fit_history/{self._name}.png")

    def _compile(
        self, hidden_neurons, dropout, activation, learning_rate, loss, branched_model, stateful=False, batch_size=32
    ):
        """Compile the model."""
        optimizer = Adam(learning_rate=learning_rate)
        # Check if multiple GPUs are available
        gpu_devices = tf.config.list_physical_devices('GPU')
        device_count = len(gpu_devices)
        if device_count > 1 and os.environ.get("USE_MULTIPLE_GPUS") == "True":
            print("Using multiple GPUs.")
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = self._create_model(hidden_neurons, dropout, activation, stateful=stateful, batch_size=batch_size)
                model.compile(loss=loss, optimizer=optimizer, metrics=["mape"])
        else:
            print("Using single GPU.")
            if(os.environ.get("USE_MULTIPLE_GPUS") == "True"):
                print("Multiple GPUs are not available.")
            else:
                print("Multiple GPUs are not enabled by environment variable.")
            model = self._create_model(hidden_neurons, dropout, activation, stateful=stateful, batch_size=batch_size)
            model.compile(loss=loss, optimizer=optimizer, metrics=["mape"])
        model.summary()
        # Plot the model
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=f"{self._path}/models/{name}.png",
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

        return model

    def compile_and_fit(
        self,
        hidden_neurons=256,
        dropout=0.4,
        activation="tanh",
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
        loss="mae",
        branched_model=False,
        patience=40,
        x_val=None,
        y_val=None,
        validation_split=0.2,
    ) -> DataFrame:
        """Compile and fit the model.

        @param hidden_neurons: The number of neurons in the hidden layers.
        @param dropout: The dropout rate between non recurrent layers.
        @param activation: The activation function for all internal nodes.
        @param epochs: The number of epochs to train the model.
        @param learning_rate: The learning rate for the optimizer.
        @param batch_size: The batch size for the training process.
        @param validation_spilt: The validation split for the training process.
        @param patience: The patience for the early stopping callback.
        @param branched_model: If True, the model will be a branched model.

        @return: The fit history.

        @remarks The metric for this model is fix and is the mean absolute percentage error (MAPE).
                 The model is saved in the checkpoints folder.
                 The validation loss is saved in the fit_history folder.
                 The tensorboard logs are saved in the tensorboard folder.
        """
        # Say how much GPU's are available
        model = self._compile(hidden_neurons, dropout, activation, learning_rate, loss, branched_model)
        # Configure callbacks (early stopping, checkpoint, tensorboard)
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
        fit = model.fit(
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
        model.load_weights(f"{self._path}/checkpoints/{self._name}_train.h5")
        self._model = model
        self._plot_fit_history(fit)
        # Convert the fit history to dataframe
        fit = DataFrame(fit.history)
        # Save the fit history
        fit.to_csv(f"{self._path}/fit_history/{self._name}.csv", index=False)
        return fit

    def predict(
        self,
        x_input: np.ndarray,
        steps=1,
        scaler: MinMaxScaler = None,
        from_saved_model=False,
    ) -> np.ndarray:
        """Predict the output for the given input.

        @param x_test: The input data for the model.
        @param from_saved_model: If True, the model will be loaded from the saved model.

        @remarks If from_saved_model is False, the model has to be fitted first.
                 The predicted values are scaled back to the original scale.
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
        #y_pred = model.predict(x_input, steps).flatten()
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
