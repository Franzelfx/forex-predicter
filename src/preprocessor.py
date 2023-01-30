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
        target: str,
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
        @param target: The column name of the y train data.
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
        self._target = target
        self._test_split = test_split
        self._time_steps_in = time_steps_in
        self._time_steps_out = time_steps_out
        self._intersection_factor = intersection_factor
        self._scale = scale
        self._time_column = time_column
        self._feature_range = feature_range
        # The train and test data
        self._train_data = None  # Input is a pandas dataframe, output is a numpy array
        self._test_data = None   # Input is a pandas dataframe, output is a numpy array
        self._scaler = dict      # A dict of scalers for each feature

        if self._test_split < 0.0 or self._test_split > 0.5:
            raise ValueError("The test split must be between 0.0 and 0.5.")
        if self._intersection_factor < 0.0 or self._intersection_factor > 0.9:
            raise ValueError("The intersection must be between 0.0 and 0.9.")
        if time_steps_in > len(data):
            raise ValueError(f"The input time steps must be smaller than the data length. {time_steps_in} > {len(data)}")
        if time_steps_in < 1:
            raise ValueError(f"The input time steps must be greater than 1. {time_steps_in} < 1")
        if time_steps_out > len(data):
            raise ValueError(f"The output time steps must be smaller than the data length. {time_steps_out} > {len(data)}")
        if time_steps_out < 1:
            raise ValueError(f"The output time steps must be greater than 1. {time_steps_out} < 1")
        # Preprocess the data concerning nan values, split and scaling
        data = self._drop_nan_rows(data)
        if self._scale:
            data = self._scale_data(data)
        self._train_data, self._test_data = self._split_train_test(data, self._test_split)

    def __str__(self) -> str:
        """Return the string representation of the preprocessors attributes in table format."""


    @property
    def x_train(self) -> np.ndarray:
        """Get the scaled x train data.

        @return: The x train data as numpy array in shape of
                  (samples, time_steps, features).
        """
        x_train = self._sliding_window(self._train_data, self._time_steps_in)
        # Remove column where nan values appear
        x_train = np.delete(x_train, self._data.columns.get_loc(self._target), axis=2)
        return x_train

    @property
    def y_train(self) -> np.ndarray:
        """Get the scaled y train data in shape of.

        @return: The y train data as numpy array in shape of
                  (samples, time).
        """
        # Y train is some sliding window with length of time_steps_out
        # and offset of time_steps_in
        y_train = self._sliding_window(self._train_data, self._time_steps_out)
        # Extract the needed colums from location of original y_train_column
        y_train = y_train[:, :, self._data.columns.get_loc(self._target)]
        # Shift by offset (n_time_steps_in)
        y_train = y_train[self._time_steps_in:, :]
        return y_train

    @property
    def x_test(self) -> np.ndarray:
        """Get the x test data for every feature.

        @return: X test as numpy array
        """
        x_test = self._sliding_window(self._test_data, self._time_steps_in)
        # Drop column with nan values
        x_test = np.delete(x_test, self._data.columns.get_loc(self._target), axis=2)
        return x_test

    @property
    def y_test(self) -> np.ndarray:
        """Get the y test data for the selected feture.

        @return: Y test as numpy array.
        """
        y_test = np.array(self._test_data[self._target])
        return y_test
    
    @property
    def target(self) -> str:
        """Get the target feature."""
        return self._target

    @property
    def scaler(self) -> dict[MinMaxScaler]:
        """Get the list of available scalers for each feature.
        
        @return: A list of scalers.

        @remarks The list is empty if the data is not scaled.
                 Furthermore, you can use the scaler to inverse the scaled data.
                 To do so, select the scaler for the feature you want to inverse
                 e.g. scaler['c'].inverse_transform(data)
        """
        return self._scaler

    def _drop_nan_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows with nan values."""
        data = data.dropna()
        return data

    def _split_train_test(self, data: pd.DataFrame, test_split:float)-> tuple:
        """Split the data into train and test set."""
        test_size = int(len(data) * test_split)
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        return train_data, test_data

    def _scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale the data."""
        # Drop date column
        data = data.drop(self._time_column, axis=1)
        # Scale the data
        scaler = []
        for column in data.columns:
            scaler.append(MinMaxScaler(feature_range=self._feature_range))
            data[column] = scaler[-1].fit_transform(data[column].values.reshape(-1, 1))
        self._scaler = dict(zip(data.columns, scaler))
        return data

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
