"""This module contains the model class for the LSTM model."""
import os
import logging
import numpy as np
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Permute,
    Multiply,
    GlobalMaxPooling1D,
    Bidirectional,
    TimeDistributed,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TimeSeriesModel:
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
    
    def _attention_layer(self, inputs, neurons):
        hidden_state = inputs[:, -1, :]
        # Calculate attention weights
        attention_weights = Dense(1, activation='tanh')(inputs)
        attention_weights = Permute([2, 1])(attention_weights)
        attention_weights = Dense(neurons, activation='softmax')(attention_weights)
        attention_weights = Permute([2, 1])(attention_weights)
        # Apply attention weights to hidden state
        attention_output = Multiply()([hidden_state, attention_weights])
        return attention_output

    def _create_model(self, input_shape, hidden_neurons: int, dropout_factor: float, activation: str) -> Sequential:
        model = Sequential()
        lstm_output = Bidirectional(LSTM(hidden_neurons, return_sequences=True))(model.input)
        attention_output = self._attention_layer(lstm_output, hidden_neurons)
        lstm2_output = Bidirectional(LSTM(round(0.5 * hidden_neurons), return_sequences=True))(attention_output)
        time_distributed_output = TimeDistributed(Dense(round(0.75 * hidden_neurons), activation=activation))(lstm2_output)
        dropout_output1 = Dropout(dropout_factor)(time_distributed_output)
        time_distributed2_output = TimeDistributed(Dense(round(0.75 * hidden_neurons), activation=activation))(dropout_output1)
        dropout_output2 = Dropout(dropout_factor)(time_distributed2_output)
        time_distributed3_output = TimeDistributed(Dense(self._y_train.shape[1], activation=activation))(dropout_output2)
        global_pooling_output = GlobalMaxPooling1D()(time_distributed3_output)
        dense_output1 = Dense(round(0.5 * hidden_neurons), activation=activation)(global_pooling_output)
        dense_output2 = Dense(round(0.5 * hidden_neurons), activation=activation)(dense_output1)
        final_output = Dense(self._y_train.shape[1], activation="linear")(dense_output2)
        model = Model(inputs=model.input, outputs=final_output)
        model.build(input_shape=input_shape)
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
        self, hidden_neurons, dropout, activation, learning_rate, loss
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
                model = self._create_model(hidden_neurons, dropout, activation)
                model.compile(loss=loss, optimizer=optimizer, metrics=["mape"])
        else:
            print("Using single GPU.")
            if(os.environ.get("USE_MULTIPLE_GPUS") == "True"):
                print("Multiple GPUs are not available.")
            else:
                print("Multiple GPUs are not enabled by environment variable.")
            model = self._create_model(hidden_neurons, dropout, activation)
            model.compile(loss=loss, optimizer=optimizer, metrics=["mape"])
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

        return model

    #TODO: Add parameter to configure model shape as lists of layers types and neurons as dict.
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
        model = self._compile(hidden_neurons, dropout, activation, learning_rate, loss)
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
