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
# Logging
from src.logger import logger as loguru

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
        self, max_lr, patience, patience_lr_schedule, log_dir, monitor="val_loss"
    ):
        super(CustomLRScheduler, self).__init__()
        self.max_lr = max_lr
        self.patience = patience
        self.patience_lr_schedule = patience_lr_schedule
        self.initial_lr = max_lr / 10
        self.warmup_epochs = patience_lr_schedule
        self.total_epochs = patience
        self.cosine_frequency = 1 / self.total_epochs
        self.log_dir = log_dir
        self.lr_writer = tf.summary.create_file_writer(log_dir + "/lr")
        self.monitor = monitor
        self.best_score = None
        self.no_improvement_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = (
                self.initial_lr
                + (self.max_lr - self.initial_lr) / self.warmup_epochs * epoch
            )
        else:
            # Use a step decay for simplicity
            decay_step = self.no_improvement_epochs // self.patience_lr_schedule
            lr = self.max_lr * (0.5 ** decay_step)  # Reduce by half every patience_lr_schedule epochs

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # Log the learning rate
        with self.lr_writer.as_default():
            tf.summary.scalar("learning_rate", lr, step=epoch)
            self.lr_writer.flush()


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
            loguru.info(f" Combined score improved to {combined_score:.4f}. Save model.")
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
        self._learning_rate = learning_rate
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

        loguru.info(f"Tensorflow version: {tf.__version__}")
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
            # loguru.info exception with traceback
            loguru.info(e)
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
        continue_training=False,  # New parameter to control continuous learning
    ) -> DataFrame:
        """Compile and fit the model.

        :param int epochs: The number of epochs for the training process.
        :param int batch_size: The batch size for the training process.
        :param int patience: The patience for the early stopping callback.
        :param float validation_split: The validation split for the training process.
        :param continue_training: If True, continue training from the last checkpoint.

        :returns: The fit history.

        """
        # Check if the model should continue training from the last checkpoint
        if continue_training:
            try:
                # Attempt to load the last saved weights
                path = f"{self._path}/checkpoints/{self._name}"
                self._model: tf.keras.Model = self.load_model(path)
                loguru.info("Continuing training from the last checkpoint.")
            except Exception as e:
                # Handle cases where loading the weights fails (e.g., no checkpoint exists)
                loguru.warning("Failed to load weights for continuous training. Starting from scratch.")

        if self._model is None:
            loguru.warning("Model is not compiled yet, please compile the model first.")
            return
        reset_states = ResetStatesCallback()
        model_checkpoint = ModelCheckpoint(
            filepath=f"{self._path}/checkpoints/{self._name}"
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", verbose=1
        )
        tensorboard = TensorBoard(log_dir=f"{self._path}/tensorboard/{self._name}")
        lr_scheduler = CustomLRScheduler(
            max_lr=self._learning_rate,
            patience=patience,
            patience_lr_schedule=patience_lr_schedule,
            log_dir=f"{self._path}/tensorboard/{self._name}",
        )
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
            loguru.error(e)
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
            loguru.info(f"Loaded model from: {path}")
        else:
            # Check if the model has been fitted
            if self._model is None:
                raise Exception(
                    "The model has not been fitted yet, please call compile_and_fit() first."
                )
            prediction_model = self._model
        # Predict the test values
        if x_test is not None:
            y_test = prediction_model.predict(x_test, use_multiprocessing=True, workers=8)
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
        # Free allocated memory from the GPU
        tf.keras.backend.clear_session()
        # Assign prediction model
        self._model = prediction_model
        # Return the predicted values, based on the given input
        return y_test, y_hat

    def sigmoid_confidence_scores(self, std_predictions):
        """
        Calculate confidence scores using a sigmoid function based on standard deviations.

        Parameters:
        - std_predictions: Standard deviations of the predictions.

        Returns:
        - confidence_scores: Confidence scores calculated using a sigmoid function, scaled to 0-100.
        """
        # Adjust the sigmoid center using median or mean standard deviation as needed
        sigmoid_center = np.median(std_predictions)

        # Sigmoid function to map standard deviation to a confidence score
        confidence_scores = 1 / (1 + np.exp(std_predictions - sigmoid_center))
        confidence_scores *= 100  # Scale to 0-100 range

        return np.clip(confidence_scores, 0, 100)  # Ensure scores are within 0-100


    def confidence(self, x_input, num_samples=50):
        """
        Calculate prediction uncertainty using Monte Carlo Dropout and return confidence percentage.

        Parameters:
        - x_input: The input data for predictions (numpy array).
        - num_samples: Number of Monte Carlo samples to generate.

        Returns:
        - mean_predictions: Mean predictions across Monte Carlo samples.
        - confidence_percent: Confidence in predictions as a percentage, calculated using a sigmoid function for smoother scaling.
        """
        predictions = []

        # Generate predictions using Monte Carlo Dropout
        for _ in range(num_samples):
            pred = self._model.predict(x_input, training=True)
            predictions.append(pred)
        predictions = np.array(predictions)

        # Calculate mean and standard deviation of predictions
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)

        # Apply sigmoid function for confidence calculation
        confidence_scores = self.sigmoid_confidence_scores(std_predictions)

        # Calculate mean confidence score across all predictions
        confidence_percent = np.mean(confidence_scores)

        return np.round(confidence_percent, 2)
