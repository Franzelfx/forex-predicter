"""Module for the Preprocessor class."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    """Used to preprocess the data for RNNs.

    @remarks The preprocessor has the following tasks:
             1. Split the data into train and test set.
             2. Scale the data.
             3. Split the data into sequences.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y_train_column: str,
        test_split=0.2,
        time_steps_in=60,
        time_steps_out=30,
        intersection_factor=0.5,
        scale=True,
        time_column='t',
        feature_range=(-1, 1),
    ):
        """Set the fundamental attributes.

        @param data: The data to be feed inside the RNN.
        @param y_train_column: The column name of the y train data.
        @param test_split: The percentage of the data to be used for testing (0.0 to 0.5).
        @param time_steps: The number of time steps to be used per sample.
        @param intersection_factor: The intersection of the single samples (0.0 to 0.9).
                                An intersection of 0 means that the samples
                                are not overlapping. An intersection of 0.9 means
                                that the samples are overlapping by 90%.
        @param scale: If the data should be scaled or not.
        @param time_column: The column name of the time column.

        @remarks The time_steps and intersection parameter will determine, how
                    much sequences will be created from the data.
        """
        # Attributes
        self._data = data
        self._y_train_column = y_train_column
        self._test_split = test_split
        self._time_steps_in = time_steps_in
        self._time_steps_out = time_steps_out
        self._intersection_factor = intersection_factor
        self._scale = scale
        self._time_column = time_column
        self._feature_range = feature_range
        # The train and test data
        self._train_data = None  # Input is a pandas dataframe, output is a numpy array
        self._test_data = None  # Input is a pandas dataframe, output is a numpy array
        self._train_dates: np.array = None
        self._test_dates: np.array = None

        if self._test_split < 0.0 or self._test_split > 0.5:
            raise ValueError("The test split must be between 0.0 and 0.5.")
        if self._intersection_factor < 0.0 or self._intersection_factor > 0.9:
            raise ValueError("The intersection must be between 0.0 and 0.9.")
        if time_steps_in > len(data):
            raise ValueError(f"The input time steps must be smaller than the data length. {time_steps_in} > {len(data)}")
        if time_steps_in < 1:
            raise ValueError(f"The input time steps must be greater than 1. {time_steps_in} < 1")
        self._split_train_test()
        if self._scale:
            self._scale_data()

    def __str__(self) -> str:
        """Return the string representation of the preprocessors attributes in table format."""


    @property
    def x_train(self) -> np.ndarray:
        """Get the scaled x train data.

        @return: The x train data as numpy array in shape of
                  (samples, time_steps, features).
        """
        x_train = self._sliding_window(self._train_data, self._time_steps_in)
        return x_train

    @property
    def y_train(self) -> np.ndarray:
        """Get the scaled y train data in shape of (samples, time_steps, features).

        @return: The y train data as numpy array in shape of
                  (samples, time).
        """
        # Y train is some sliding window with length of time_steps_out
        # and offset of time_steps_in
        y_train = self._sliding_window(self._train_data, self._time_steps_out)
        # Extract the needed colums from location of original y_train_column
        y_train = y_train[:, :, self._train_data.columns.get_loc(self._y_train_column)]
        # Shift by offset (n_time_steps_in)
        y_train = y_train[self._time_steps_in:, :]
        # Reshape to (samples, time_steps, features)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        return y_train

    @property
    def x_test(self) -> np.ndarray:
        """Get the x test data for every feature.

        @return: X test as numpy array
        """
        x_test = np.array(self._test_data)
        return x_test

    @property
    def y_test(self) -> np.ndarray:
        """Get the y test data for the selected feture.

        @return: Y test as numpy array.
        """
        y_test = np.array(self._test_data[self._y_train_column])
        return y_test

    def _split_train_test(self):
        """Split the data into train and test set."""
        test_size = int(len(self._data) * self._test_split)
        self._train_data = self._data[:-test_size]
        self._test_data = self._data[-test_size:]

    def _scale_data(self):
        """Scale the data."""
        scaler = MinMaxScaler(feature_range=self._feature_range)
        # Temporaryly drop the date column
        self. _train_data = self._train_data.drop(self._time_column, axis=1)
        self._test_data = self._test_data.drop(self._time_column, axis=1)
        self._train_data = scaler.fit_transform(self._train_data)
        self._test_data = scaler.transform(self._test_data)
        # Recover the date column
        self._train_data = pd.DataFrame(self._train_data, columns=self._data.columns[:-1])
        self._test_data = pd.DataFrame(self._test_data, columns=self._data.columns[:-1])

    def _sliding_window(self, input_sequence: pd.DataFrame, time_steps: int, offset: int = 0):
        """Sliding window with m time steps and n features (given by the __init__ parameters).

        @param input_sequence: The input sequence to be split into samples.
        @return: The samples as numpy array.

        @remarks The time_steps and intersection parameter will determine, how
                    much sequences will be created from the data.
                    (the higher the intersection, the more sequences)
        """
        intersection_length = int(time_steps * self._intersection_factor)
        samples = []
        for i in range(0, len(input_sequence) - time_steps + offset, time_steps - intersection_length):
            sample = input_sequence[i : i + time_steps]
            samples.append(sample)
        samples = np.array(samples) # shape: (samples, time_steps, features)
        return samples
