import os
import math
import logging
import numpy as np
from typing import List
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from datetime import datetime as dt
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau,
)
import src.layers as layers
from tensorflow.keras.layers import Concatenate, LayerNormalization, Input
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Branch:
    def __init__(
        self,
        _input: np.ndarray,
        transformer_neurons,
        lstm_neurons,
        dense_neurons,
        attention_heads,
        dropout_rate,
    ):
        self.input = _input
        self.transformer_neurons = transformer_neurons
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


class CustomLRScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        warmup_epochs = 10,
        initial_lr = 0.0001,
        max_lr = 0.0005,
        final_lr = 0.00001,
        total_epochs = 1000,
        cosine_frequency = 10,
    ):
        super(CustomLRScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs
        self.cosine_frequency = cosine_frequency

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = (
                self.initial_lr
                + (self.max_lr - self.initial_lr) / self.warmup_epochs * epoch
            )
        else:
            decayed_lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (
                1
                + math.cos(
                    math.pi
                    * self.cosine_frequency
                    * (epoch - self.warmup_epochs)
                    / (self.total_epochs - self.warmup_epochs)
                )
            )
            lr = max(decayed_lr, self.final_lr)

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_best_only=True):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.best_score = float("inf")  # Initialize with a high value
        self.save_best_only = save_best_only

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        val_mape = logs.get("val_mape")

        # Scale the metrics
        scaled_loss = val_loss
        scaled_mape = (
            val_mape / 100.0
        )  # Scale down the mape value by dividing by a factor

        # Define the combined score (you can use any formula that suits your needs)
        combined_score = scaled_loss + scaled_mape

        # Check if the combined score is better than the best score seen so far
        if combined_score < self.best_score:
            self.best_score = combined_score
            print(f" Combined score improved to {combined_score:.4f}. Save model.")
        if self.save_best_only:
            self.model.save(self.filepath)
        else:
            filepath = self.filepath + f"_{epoch:04d}_{combined_score:.4f}"
            self.model.save(filepath)  # Save the model in .keras format


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
        self._architecture = None
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
            _branch = layers.Branch(
                branch.transformer_neurons,
                branch.lstm_neurons,
                branch.dense_neurons,
                branch.attention_heads,
                branch.dropout_rate,
            )(_input)
            branches.append(_branch)
            inputs.append(_input)
        # Summation layer
        if len(branches) > 1:
            summation = Concatenate()(branches)
        # Layer normalization
        if len(branches) > 1:
            summation = LayerNormalization()(summation)
        else:
            summation = LayerNormalization()(branches[0])
        # Main branch after the summation
        main_branch = layers.Branch(
            architecture.main_branch.transformer_neurons,
            architecture.main_branch.lstm_neurons,
            architecture.main_branch.dense_neurons,
            architecture.main_branch.attention_heads,
            architecture.main_branch.dropout_rate,
        )(summation)
        output = layers.Output(
            architecture.output.hidden_neurons,
            architecture.output.dropout_rate,
            self._y_train.shape[1],
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
        self._architecture = architecture
        optimizer = Adam(learning_rate=learning_rate)
        # Metrics
        mae = MeanAbsoluteError(name="mae")
        mse = MeanSquaredError(name="mse")
        metrics_list = ["mape", mae, mse]

        # Check if multiple GPUs are available
        if strategy is not None and hasattr(strategy, "scope"):
            with strategy.scope():
                model = self._build(architecture)
                model.compile(loss=loss_fct, optimizer=optimizer, metrics=metrics_list)
        else:
            model = self._build(architecture)
            model.compile(loss=loss_fct, optimizer=optimizer, metrics=metrics_list)

        print(f"Tensorflow version: {tf.__version__}")
        model.summary(expand_nested=True)
        # Plot the model
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=f"{self._path}/models/{self._name}.png",
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=300,
            )
        except Exception as e:
            # Print exception with traceback
            print(e)
            logging.error(e)
        self._model = model

    def fit(
        self,
        epochs=1000,
        batch_size=32,
        patience=250,
        patience_lr_schedule=100,
        validation_split=0.1,
        strategy=None,
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
            filepath=f"{self._path}/checkpoints/{self._name}"
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", verbose=1
        )
        tensorboard = TensorBoard(log_dir=f"{self._path}/tensorboard/{self._name}")
        # lr_scheduler = ReduceLROnPlateau(
        #    factor=0.5, patience=patience_lr_schedule, min_lr=1e-10
        # )
        # TODO: pass the parameters to the scheduler
        lr_scheduler = CustomLRScheduler()
        # Split the data
        X_train = []
        X_val = []
        for branch in self._architecture.branches:
            _X_train, _X_val = train_test_split(
                branch.input, test_size=validation_split, shuffle=False
            )
            X_train.append(_X_train)
            X_val.append(_X_val)
        y_train, y_val = train_test_split(
            self._y_train, test_size=validation_split, shuffle=False
        )
        # Fit the model
        try:
            if strategy is not None and hasattr(strategy, "scope"):
                with strategy.scope():
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
            else:
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
            self._model.load_weights(f"{self._path}/checkpoints/{self._name}")
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

    def load_model(self, path) -> tf.keras.Model:
        model = load_model(
            path,
            custom_objects={
                "LSTM": tf.keras.layers.LSTM,
                "TransformerBlock": layers.TransformerBlock,
                "TransformerLSTMBlock": layers.TransformerLSTMBlock,
                "Branch": layers.Branch,
                "Output": layers.Output,
            },
        )
        return model

    def predict(
        self,
        x_hat: np.ndarray,
        scaler: StandardScaler = None,
        from_saved_model=True,
        x_test: np.ndarray = None,
    ) -> np.ndarray:
        """Predict the output for the given input.
        :param np.ndarray x_input: The input data for the prediction as numpy array with shape (samples, time_steps, features).
        :param int steps: The number of steps to predict (one step is all steps for one sample).

        :remarks:   • If from_saved_model is False, "compile_and_fit()" must be called first.
                    • The predicted values are scaled back to the original scale if a scaler is given.
        """
        path = f"{self._path}/checkpoints/{self._name}"

        # Get the model
        if from_saved_model:
            prediction_model: tf.keras.Model = self.load_model(path)
            print(f"Loaded model from: {path}")
        else:
            # Check if the model has been fitted
            if self._model is None:
                raise Exception(
                    "The model has not been fitted yet, please call compile_and_fit() first."
                )
            prediction_model = self._model
        # Predict the test values
        if x_test is not None:
            y_test = prediction_model.predict(x_test)
            prediction_model.reset_states()
        else:
            y_test = None

        # Predict the future values
        y_hat = prediction_model.predict(x_hat).flatten()

        # Scale the output back to the original scale
        if scaler is not None:
            y_hat = self._inverse_transform(scaler, y_hat)
            if x_test is not None:
                y_test = self._inverse_transform(scaler, y_test)
        # Return the predicted values, based on the given input
        return y_test, y_hat

    def confidence(
        self,
        x_input,
        num_samples=10,
        from_saved_model=True,
    ):
        """
        Calculate prediction uncertainty using Monte Carlo Dropout and return confidence percentage.

        Parameters:
        - model: The compiled Keras model with dropout layers.
        - x_input: The input data for predictions (numpy array).
        - num_samples: Number of Monte Carlo samples to generate.
        - from_saved_model: Whether to load the model from a saved file (True) or use the provided model (False).
        - std_bound: The upper bound for standard deviation beyond which confidence is 0%.

        Returns:
        - mean_predictions: Mean predictions across Monte Carlo samples.
        - confidence_percent: Confidence in predictions as a percentage.
        """

        path = f"{self._path}/checkpoints/{self._name}"

        if from_saved_model:
            # Load the model from a saved file
            prediction_model = load_model(
                path,
                custom_objects={
                    "LSTM": tf.keras.layers.LSTM,
                    "TransformerBlock": layers.TransformerBlock,
                    "TransformerLSTMBlock": layers.TransformerLSTMBlock,
                    "Branch": layers.Branch,
                    "Output": layers.Output,
                },
            )
        else:
            # Check if the model has been compiled
            if self._model is None:
                raise ValueError("The model must be compiled first.")

            prediction_model = self._model

        # Initialize arrays to store predictions from Monte Carlo samples
        predictions = []

        # Generate predictions using Monte Carlo Dropout
        for _ in range(num_samples):
            predictions.append(prediction_model.predict(x_input))

        # Calculate mean and standard deviation of predictions
        std_predictions = np.std(predictions, axis=0)

        # Dynamic scaling using median (or mean) of the std_predictions
        median_std = np.median(std_predictions)

        # Guard against division by zero
        if median_std == 0:
            median_std = 1e-10  # small constant to avoid division by zero

        confidence_percent = np.clip(100 * (1 - (std_predictions / median_std)), 0, 100)

        # Calculate mean of confidence across all samples as 2 decimal percentage
        confidence_percent = np.mean(confidence_percent)
        confidence_percent = np.round(confidence_percent, 2)

        return confidence_percent
