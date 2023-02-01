"""This module contains the model class for the LSTM model."""
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class Model:
    """Used to create, compile, fit and predict with the LSTM model."""

    def __init__(
        self,
        path: str,
        name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dropout: float = 0.2,
        loss: str = "mean_squared_error",
        metrics: list = ["mape"],
    ):
        """Set the fundamental attributes.

        @param path: The top level path to the checkpoint, fit, model and tensorboard folder.
        @param name: The name of the model (e.g. EURUSD in case of finacial analysis).
        @param x_train: The input data for the model.
        @param y_train: The output data for the model.
        @param dropout: The dropout rate in the layer with the most neurons.
        @param loss: The loss function for the training process.
        @param optimizer: The optimizer for the training process.
        @param metrics: The metrics to measure the performance of the model.
        """
        self._path = path
        self._name = name
        self._x_train = x_train
        self._y_train = y_train
        self._dropout = dropout
        self._loss = loss
        self._metrics = metrics
        self._model = None

    def _create_model(self, hidden_neurons=128) -> Sequential:
        """Create the model."""
        model = Sequential()
        model.add(Bidirectional(LSTM(hidden_neurons, return_sequences=True, input_shape=(self._x_train.shape[1], self._x_train.shape[2]))))
        model.add(Bidirectional(LSTM(hidden_neurons, return_sequences=True)))
        model.add(Bidirectional(LSTM(hidden_neurons, return_sequences=False)))
        model.add(Dense(hidden_neurons))
        model.add(Dropout(self._dropout))
        model.add(Dense(hidden_neurons))
        model.add(Dropout(self._dropout))
        model.add(Dense(self._y_train.shape[1]))
        model.build(input_shape=(self._x_train.shape[0], self._x_train.shape[1], self._x_train.shape[2]))
        return model

    def _plot_fit_history(self, fit):
        """Plot the fit history."""
        # High resolution plot with subplots
        plt.cla()
        plt.clf()
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        # Plot the loss
        axes[0].plot(fit.history["loss"], label="loss")
        axes[0].plot(fit.history["val_loss"], label="val_loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Loss")
        axes[0].legend()
        # Plot the metrics
        axes[1].plot(fit.history["mape"], label="mape")
        axes[1].plot(fit.history["val_mape"], label="val_mape")
        axes[1].set_ylabel("MAPE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("MAPE")
        axes[1].legend()
        # Turn on the grid
        axes[0].grid()
        axes[1].grid()
        fig.tight_layout()
        # Save the plot
        plt.savefig(f"{self._path}/fit_history/{self._name}.png")

    def compile_and_fit(self, hidden_neurons=256, epochs=100, learning_rate=0.001, batch_size=32, validation_spilt=0.2, patience=20) -> dict:
        """Compile and fit the model."""
        model = self._create_model(hidden_neurons)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss=self._loss, optimizer=optimizer, metrics=self._metrics)
        model.summary()
        # Configure callbacks (early stopping, checkpoint, tensorboard)
        model_checkpoint = ModelCheckpoint(
            filepath=f"{self._path}/checkpoints/{self._name}_weights.h5",
            monitor="val_mape",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        early_stopping = EarlyStopping(monitor="val_mape", patience=patience, mode="min", verbose=1)
        tensorboard = TensorBoard(log_dir=f"{self._path}/tensorboard/{self._name}")
        # Fit the model
        fit = model.fit(
            self._x_train,
            self._y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_spilt,
            callbacks=[model_checkpoint, early_stopping, tensorboard],
            shuffle=False,
        )
        # Load the best weights
        model.load_weights(f"{self._path}/checkpoints/{self._name}_weights.h5")
        self._model = model
        self._plot_fit_history(fit)
        return fit

    def predict(self, x_test: np.ndarray, scaler:MinMaxScaler=None, from_saved_model=False) -> np.ndarray:
        """Predict the output for the given input.
        
        @param x_test: The input data for the model.
        @param from_saved_model: If True, the model will be loaded from the saved model.

        @remarks If from_saved_model is False, the model has to be fitted first.
                 The predicted values are scaled back to the original scale.
        """
        if from_saved_model:
            model = self._create_model()
            model.load_weights(f"{self._path}/checkpoints/{self._name}_weights.h5")
        else:
            # Check if the model has been fitted
            if self._model is None:
                raise Exception(
                    "The model has not been fitted yet, plase call compile_and_fit() first."
                )
            model = self._model
        # Predict the output
        y_pred = model.predict(x_test).flatten()
        y_pred = y_pred.reshape(-1, 1)
        if scaler is not None:
            y_pred = scaler.inverse_transform(y_pred)
        return y_pred
