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
from sklearn.preprocessing import StandardScaler
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


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()


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
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 1,
    ):
        # Check if name has ":" in it, if so get characters after it
        if ":" in name:
            name = name.split(":")[1]
        self._name = name
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
        self._path = path
        self._x_train = x_train
        self._y_train = y_train
        self._batch_size = batch_size
        self._model = None

    @property
    def steps_ahead(self) -> int:
        """Get the steps ahead."""
        return self._y_train.shape[1]

    def _var_name(self, var):
        variable_names = [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]
        if variable_names:
            return variable_names[0]
        else:
            return None
    
    def _inverse_transform(self, scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
        """Inverse transform the data."""
        data = data.reshape(-1, 1)
        data = scaler.inverse_transform(data).flatten()
        return data
    
    def _transformer_block(self, hidden_neurons, attention_heads, dropout_rate, input_tensor):
        input_matched_1 = Dense(hidden_neurons)(input_tensor)

        # Separate query and value branches for Attention layer
        query = Dense(hidden_neurons)(input_matched_1)
        value = Dense(hidden_neurons)(input_matched_1)

        # Apply Attention layer
        attention_1 = MultiHeadAttention(attention_heads, hidden_neurons)(query, value)

        # Add dropout and residual connection and layer normalization
        dropout_attention = Dropout(dropout_rate)(attention_1)
        residual_attention = Add()([input_matched_1, dropout_attention])
        norm_attention = LayerNormalization()(residual_attention)

        # Feed forward layer
        feed_forward_1 = Dense(hidden_neurons, activation="relu")(norm_attention)
        feed_forward_2 = Dense(hidden_neurons, activation="relu")(feed_forward_1)
        feed_forward_3 = Dense(hidden_neurons, activation="relu")(feed_forward_2)

        # Add dropout, residual connection, and layer normalization
        dropout_ffn = Dropout(dropout_rate)(feed_forward_3)
        residual_ffn = Add()([norm_attention, dropout_ffn])
        norm_ffn = LayerNormalization()(residual_ffn)

        return norm_ffn


    def _build(self, hidden_neurons: int, dropout_rate: float, attention_heads: int, key_dim=16):
        input_shape = (self._x_train.shape[1], self._x_train.shape[2])
        inputs = Input(batch_shape=(self._batch_size,) + input_shape)

        # Transformer Block 1
        transformer_block_1 = self._transformer_block(
            hidden_neurons, attention_heads, dropout_rate, inputs
        )
        # LSTM Block 1
        lstm_1 = Bidirectional(LSTM(hidden_neurons, return_sequences=True))(inputs)
        # Match LSTM output shape to Transformer Block output shape
        lstm_matched_1 = Dense(hidden_neurons)(lstm_1)
        # Add and normalize
        add_1 = Add()([transformer_block_1, lstm_matched_1])
        norm_1 = LayerNormalization()(add_1)

        # Transformer Block 2
        transformer_block_2 = self._transformer_block(
            hidden_neurons, attention_heads, dropout_rate, norm_1
        )
        # LSTM Block 2
        lstm_2 = Bidirectional(LSTM(hidden_neurons, return_sequences=True))(norm_1)
        # Match LSTM output shape to Transformer Block output shape
        lstm_matched_2 = Dense(hidden_neurons)(lstm_2)
        # Add and normalize
        add_2 = Add()([transformer_block_2, lstm_matched_2])
        norm_2 = LayerNormalization()(add_2)

        # Global average pooling
        gap = GlobalAveragePooling1D()(norm_2)

        # Dense layers
        dense_1 = Dense(hidden_neurons, activation="relu")(gap)
        dropout_3 = Dropout(dropout_rate)(dense_1)
        dense_2 = Dense(hidden_neurons, activation="relu")(dropout_3)
        dropout_4 = Dropout(dropout_rate)(dense_2)
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
        dropout_rate: float = 0.2,
        attention_heads: int = 2,
        loss_fct: str = "mae",
        strategy=None,
    ):
        """Compile the model."""
        optimizer = Adam(learning_rate=learning_rate)
        # Check if multiple GPUs are available
        if strategy is not None and hasattr(strategy, "scope"):
            with strategy.scope():
                model = self._build(hidden_neurons, dropout_rate, attention_heads)
                model.compile(loss=loss_fct, optimizer=optimizer, metrics=["mape"])
        else:
            model = self._build(hidden_neurons, dropout_rate, attention_heads)
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
        x_val=None,
        y_val=None,
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
                validation_data=(x_val, y_val)
                if (x_val and y_val) is not None
                else None,
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
        #BUG: wrong vstacking
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
