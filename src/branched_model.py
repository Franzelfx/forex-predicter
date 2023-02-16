"""Branched model"""
import numpy as np
from logging import warning
from pandas import DataFrame
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    concatenate,
)

class Branched_Model:

    def __init__(self, model_path: str, pair: str, x_train: dict, y_train: np.ndarray):
        """Initialize the model."""
        self.model_path = model_path
        self.pair = pair
        self._x_train = x_train
        self._y_train = y_train
        self._branches = None

    def _create_model(
        self,
        conv=[] or None,
        lstm=[] or None,
        dense=[64],
        dropout=0.2,
        activation="tanh",
    ) -> Sequential:
        """Create the branched model."""
        model = Sequential()
        if self._branches is None:
            raise ValueError("Please set the model branches first.")
        model = concatenate(self._branches)
        # Add output layer(s)
        output = self.add_branch(
            conv=conv, lstm=lstm, dense=dense, dropout=dropout, activation=activation
        )
        # Get inputs
        inputs = []
        for branch in self._branches:
            inputs.append(branch.input)
        model = Model(inputs=inputs, outputs=output.output)
        model.summary()
        return model

    def add_branch(
        self,
        x_train:np.array,
        y_train:np.array,
        conv=[] or None,
        lstm=[] or None,
        dense=[64],
        dropout=0.2,
        activation="tanh",
        kernel_size=3,
        pool_size=2,
    ):
        """Add a branch to the branched model."""
        if self._branches is None:
            self._branches = []
        model = Sequential()
        if conv is not None:
            for i in range(len(conv)):
                if i == 0:
                    model.add(
                        Conv1D(
                            conv[i],
                            kernel_size,
                            activation=activation,
                            input_shape=(
                                x_train.shape[1],
                                x_train.shape[2],
                            ),
                        )
                    )
                else:
                    model.add(Conv1D(conv[i], kernel_size, activation=activation))
                model.add(MaxPooling1D(pool_size=pool_size))
        if lstm is not None:
            for i in range(len(lstm)):
                if i == 0:
                    model.add(
                        LSTM(
                            lstm[i],
                            return_sequences=True,
                            input_shape=(
                                x_train.shape[1],
                                x_train.shape[2],
                            ),
                        )
                    )
                else:
                    model.add(LSTM(lstm[i], return_sequences=True))
                model.add(Dropout(dropout))
        model.add(Dropout(dropout))
        for i in range(len(dense)):
            model.add(Dense(dense[i], activation=activation))
        model.add(Dense(y_train.shape[1]))

    def compile_and_fit(self):
        self._model = self._create_model()
        return self._model