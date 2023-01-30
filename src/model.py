"""This module contains the model class for the LSTM model."""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
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
        optimizer: str = "adam",
        metrics: list = ["accuracy"],
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
        self._optimizer = optimizer
        self._metrics = metrics
        self._model = None

    def _create_model(self, hidden_neurons=50) -> Sequential:
        """Create the model."""
        model = Sequential()
        model.add(
            LSTM(
                hidden_neurons,
                input_shape=(self._x_train.shape[1], self._x_train.shape[2]),
            )
        )
        model.add(Dropout(self._dropout))
        model.add(Dense(self._y_train.shape[1]))
        return model

    def _plot_fit_history(self, fit):
        """Plot the fit history."""
        # High resolution plot
        plt.figure(figsize=(20, 10))
        plt.plot(fit.history["loss"], label="train")
        plt.plot(fit.history["val_loss"], label="test")
        plt.legend()
        plt.savefig(f"{self._path}/fit_history/{self._name}.png")

    def compile_and_fit(self, hidden_neurons=50, epochs=100, batch_size=32, validation_spilt=0.2, patience=10) -> dict:
        """Compile and fit the model."""
        model = self._create_model(hidden_neurons)
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        model.summary()
        # Configure callbacks (early stopping, checkpoint, tensorboard)
        model_checkpoint = ModelCheckpoint(
            filepath=f"{self._path}/checkpoints/{self._name}_weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience)
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
        # Save the model
        model.save(f"{self._path}/models/{self._name}.h5")
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
