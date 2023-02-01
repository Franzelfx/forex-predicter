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
        data = self._drop_nan(data)
        if self._scale:
            data = self._scale_data(data)
        # Data is now without time and index column
        self._data = data
        self._train_data, self._test_data = self._split_train_test(data, self._test_split)
        self._x_train, self._y_train = self._create_samples(self._train_data, self._time_steps_in, self._time_steps_out)

    def __str__(self) -> str:
        """Return the string representation of the preprocessors attributes in table format."""

    @property
    def data(self) -> pd.DataFrame:
        """Get the original data but with removed nan values.

        @return: The preprocessed data as pandas dataframe.
        @remarks The data is without the time and index column
                 (removed by the _drop_nan method).
        """
        return self._data

    @property
    def x_train(self) -> np.ndarray:
        """Get the scaled x train data.

        @return: The x train data as numpy array in shape of
                  (samples, time_steps, features).
        """
        return self._x_train

    @property
    def y_train(self) -> np.ndarray:
        """Get the scaled y train data in shape of.

        @return: The y train data as numpy array in shape of
                  (samples, time).
        """
        # Y train is some sliding window with length of time_steps_out
        # and offset of time_steps_in
        return self._y_train

    @property
    def x_test(self) -> np.ndarray:
        """Get the x test data for every feature.

        @return: X test as numpy array
        """
        x_test = self._test_data.values[-self._time_steps_in:]
        # Reshape the test data to (samples, time_steps, features)
        x_test = np.reshape(x_test, (1, self._time_steps_in, x_test.shape[1]))
        return x_test

    @property
    def y_test(self) -> np.ndarray:
        """Get the y test data for the selected feture.

        @return: Y test as numpy array.
        """
        y_test = np.array(self._test_data[self._target])
        # Inverse the scaled data
        if self._scale:
            # Reshape the test data
            y_test = y_test.reshape(-1, 1)
            y_test = self._scaler[self._target].inverse_transform(y_test)
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
    
    @property
    def time_steps_in(self) -> int:
        """Get the number of time steps for the input."""
        return self._time_steps_in

    @property
    def time_steps_out(self) -> int:
        """Get the number of time steps for the output."""
        return self._time_steps_out

    def _drop_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows with nan values."""
        # Safe the header
        header = data.columns
        # Get location of columns in which all values are nan
        nan_columns = data.isna().all()
        nan_rows = data.isna().any(axis=1)
        # Drop nan columns and rows
        data = data.drop(data.columns[nan_columns], axis=1)
        data = data.drop(data.index[nan_rows])
        # Drop the indedx column
        data = data.drop(data.columns[0], axis=1)
        # Update the header
        header = header.drop(header[nan_columns])
        header = header.drop(header[0])
        # Add header to data
        data.columns = header
        # Check if target column is still in data
        if self._target not in header:
            raise ValueError(f"The target column {self._target} is not in the data anymore.")
        return data

    def _split_train_test(self, data: pd.DataFrame, test_split:float)-> tuple:
        """Split the data into train and test set."""
        test_size = int(len(data) * test_split)
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        return train_data, test_data

    def _scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale the data."""
        # Scale the data
        scaler = []
        for column in data.columns:
            scaler.append(MinMaxScaler(feature_range=self._feature_range))
            data[column] = scaler[-1].fit_transform(data[column].values.reshape(-1, 1))
        self._scaler = dict(zip(data.columns, scaler))
        return data

    def _create_samples(self, input_sequence: pd.DataFrame, steps_in: int, steps_out: int) -> tuple:
        """Create samples of x and y.
        
        @param input_sequence: The input sequence.
        @param steps_in: The number of time steps fed into the network.
        @param steps_out: The number of time steps the network predicts.
        
        @return: A tuple of x and y samples.
        
        @remarks: The x samples are of shape (samples, time_steps, features).
                  The y samples are of shape (samples, time_steps).
                  After every sample of x there is a sample of y.
        """
        x = []
        y = []
        iterator = 0
        # x is simply a sliding window of length steps_in
        # y is a sliding window of length steps_out
        # with an offset of steps_in
        while iterator + steps_in + steps_out <= len(input_sequence):
            x.append(input_sequence[iterator:iterator + steps_in].values)
            y.append(input_sequence[iterator + steps_in:iterator + steps_in + steps_out][self._target].values)
            iterator += steps_in
        return np.array(x), np.array(y)