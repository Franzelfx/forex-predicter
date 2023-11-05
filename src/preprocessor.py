"""Module for the Preprocessor class."""
import numpy as np
import pandas as pd
from logging import warning
from tabulate import tabulate
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import StandardScaler


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
        time_steps_in=60,
        time_steps_out=30,
        test_length=90,
        target: str = None,
        test_split: float = None,
        scale: bool = True,
        shift: int = None,
        overlap: int = 0,
    ):
        """Set the fundamental attributes.

        @param data: The data to be feed inside the RNN.
        @param target: The column name of the y train data.
        @param time_steps_in: The number of time steps for the model input.
        @param time_steps_out: The number of time steps for the model output.
        @param test_length: The length of the test data (will be ignored if test_split is not None).
        @param test_split: The percentage of the data to be used for testing (0.0 to 0.5).
        @param scale: If the data should be scaled or not.
        @param overlap: The amount of overlap for x and y samples.
        @param prediction_mode: If the preprocessor is used for prediction or not.
                                In prediction mode, no y samples are generated
                                and there are basically only samples of x.
                                To use it for prediction, take x_test
                                as input for the model.

        @remarks The preprocessor has the following tasks:
                    1. Scale the data.
                    2. Split into train and test set.
                    3. Create samples for x and y.
        """
        # Data and target
        self._data = data
        self._target = target
        # The time steps and overlap
        self._overlap = overlap
        self._time_steps_in = time_steps_in
        self._time_steps_out = time_steps_out
        # The test length
        self._test_split = test_split
        self._test_length = test_length
        # The scaling and feature range
        self._scale = scale
        self._shift = shift
        # The train and test data
        self._train_data = None  # Input is a pandas dataframe, output is a numpy array
        self._test_data = None  # Input is a pandas dataframe, output is a numpy array
        self._scaler = dict  # A dict of scalers for each feature

        self._hat_length = time_steps_out

        # Data and target
        if self._target is not None and self._target not in data.columns:
            raise ValueError(
                f"The target column {self._target} is not present in the data."
            )
        # The time steps
        if time_steps_in > len(data):
            raise ValueError(
                f"The input time steps must be smaller than the data length. [{time_steps_in} > {len(data)}]"
            )
        if time_steps_in < 1:
            raise ValueError(
                f"The input time steps must be greater than 1. [{time_steps_in} < 1]"
            )
        if time_steps_out > len(data):
            raise ValueError(
                f"The output time steps must be smaller than the data length. [{time_steps_out} > {len(data)}]"
            )
        if time_steps_out < 1:
            raise ValueError(
                f"The output time steps must be greater than 1. [{time_steps_out} < 1]"
            )
        # The test length
        if self._test_length > 1 / 2 * len(data):
            raise ValueError(
                f"The test length must be smaller than 1/2 of the data length. [{test_length} > {1/2 * len(data)}]"
            )
        if self._test_length < time_steps_in + time_steps_out:
            self._test_length = time_steps_in + time_steps_out
            warning(f"Test length set to {self._test_length}]")
        # The test split
        if test_split is not None:
            if test_split < 0.0 or test_split > 0.5:
                raise ValueError(
                    f"The test split must be between 0.0 and 0.5. [{test_split} < 0.0 or {test_split} > 0.5]"
                )
        # The shift
        if shift is not None:
            if shift < 0:
                raise ValueError(f"The shift must be greater than 0. [{shift} < 0]")
            if time_steps_out % shift != 0:
                warning(
                    "Shift does not divide time_steps_out. (May lead to unexpected results)"
                )
        # Drop nan, if necessary
        self._data = self._drop_nan(data)
        # Get last known x and y value
        if self._target is not None:
            self._last_known_y = self._data[self._target].iloc[-1]
            self._first_known_y = self._data[self._target].iloc[-self._time_steps_out]
        # Scale the data
        if self._scale:
            self._data = self._scale_data(self._data)
        # Split the data into train and test set
        self._train_data, self._test_data = self._split_train_test(
            self._data, self._test_split, self._test_length
        )
        # Split the train data into sequences
        self._x_train, self._y_train = self._create_samples(
            self._train_data,
            self._time_steps_in,
            self._time_steps_out,
            self._overlap,
        )
        # Split the test data into sequences
        self._x_test, self._y_test = self._create_samples(
            self._test_data,
            self._time_steps_in,
            self._time_steps_out,
            self._overlap,
        )
        if self._y_test is None:
            raise ValueError("y_test is None. (May be caused by overlap > time_steps_in) or (May be caused by test_split > 0.5) or (May be caused by test_length < time_steps_in + time_steps_out)")

    def summary(self) -> None:
        """Print a summary of the preprocessor."""
        print("Preprocessor:")
        print(self._data.head(5))
        print(self.__str__())

    def __str__(self) -> str:
        """Return the string representation of the preprocessor."""
        header = ["Data", "Shape", "Size", "Remarks"]
        data = [
            ["Input", self._data.shape, len(self._data), "(Timesteps, Features)"],
            [
                "Train",
                self._train_data.shape,
                len(self._train_data),
                "(Timesteps, Features)",
            ],
            [
                "Test",
                self._test_data.shape,
                len(self._test_data),
                "(Timesteps, Features)",
            ],
            [
                "X Train",
                self._x_train.shape,
                len(self._x_train),
                "(Samples, Timesteps, Features)",
            ],
            [
                "Y Train",
                self._y_train.shape,
                len(self._y_train),
                "(Samples, Timesteps)",
            ],
            [
                "X Test",
                self._x_test.shape,
                len(self._x_test),
                "(Samples, Timesteps, Features)",
            ],
            ["Y Test", self._y_test.shape, len(self._y_test), "(Samples, Timesteps)"],
            [
                "X Hat",
                self.x_hat.shape,
                len(self.x_hat),
                "(Samples, Timesteps, Features)",
            ],
        ]
        return tabulate(data, headers=header, tablefmt="rst")

    @property
    def scale(self) -> bool:
        """Get the scale attribute.

        @return: The scale attribute as boolean.
        """
        return self._scale

    @property
    def data(self) -> pd.DataFrame:
        """Get the original data but with removed nan values.

        @return: The preprocessed data as pandas dataframe.
        @remarks The data is without the time and index column
                 (removed by the _drop_nan method).
        """
        return self._data

    @property
    def header(self) -> list:
        """Get the header of the data.

        @return: The header of the data as list.
        """
        return self._data.columns.tolist()

    @property
    def validation_split(self) -> float:
        """Get the validation split.

        @return: The validation split as float.
        @remars: val_split is (1 / samples of y_train) * time_steps_out
        """
        return (1 / self._y_train.shape[0]) * self._time_steps_out

    @property
    def test_split(self) -> float:
        """Get the test split.

        @return: The test split as float.
        """
        return self._test_split

    @property
    def train_data(self) -> pd.DataFrame:
        """Get the train data.

        @return: The train data as pandas dataframe.
        """
        return self._train_data

    @property
    def test_data(self) -> pd.DataFrame:
        """Get the test data.

        @return: The test data as pandas dataframe.
        """
        return self._test_data

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
        return self._y_train

    @property
    def x_test(self) -> np.ndarray:
        """Get the x test data for every feature.

        @return: Last sample of x_test as numpy array in shape of (timesteps, features).
        """
        return self._x_test

    @property
    def x_test_target_inverse(self) -> np.ndarray:
        """Get the x test data for the selected feture.

        @return: Last sample of x_train as numpy array in shape of (timesteps, features).
        """
        if self._target is None:
            raise ValueError("Target is not set")
        x_test = self._x_test[:, :, self.loc_of(self._target)].flatten()
        return self.target_scaler.inverse_transform(x_test.reshape(-1, 1)).flatten()

    @property
    def y_test(self) -> np.ndarray:
        """Get the y test data for the selected feture.

        @return: Y test as numpy array in shape of (samples, timesteps).
        @remarks The y test data is the target feature
                    shifted by the time steps in.
        """
        return self._y_test

    @property
    def y_test_inverse(self) -> np.ndarray:
        """Get the inverse scaled y test data.

        @return: The inverse scaled y test data as numpy array.
        """
        return self.target_scaler.inverse_transform(
            self._y_test.reshape(-1, 1)
        ).flatten()

    @property
    def x_hat(self) -> np.ndarray:
        """Get x_predict (last n_steps_in of data)"""
        x_predict = self._data[-self._time_steps_in :].values
        # Add samples dimension
        x_predict = np.expand_dims(x_predict, axis=0)
        return x_predict

    @property
    def x_hat_target_inverse(self) -> np.ndarray:
        """Get x_predict (last n_steps_in of data)"""
        if self._target is None:
            raise ValueError("Target is not set")
        x_predict = self._data[-self._time_steps_in :][self._target].values
        return self.target_scaler.inverse_transform(x_predict.reshape(-1, 1)).flatten()

    @property
    def first_known_y(self) -> np.ndarray:
        """Get the last known value for each feature.

        @return: The last known value of x_test as numpy array.
        """
        return self._first_known_y

    @property
    def last_known_y(self) -> int:
        """Get the last known value for the target feature.

        @return: The last known value of y_test as integer.
        """
        return self._last_known_y

    @property
    def target(self) -> str:
        """Get the target feature."""
        return self._target

    @property
    def scaler(self) -> dict[StandardScaler]:
        """Get the list of available scalers for each feature.

        @return: A list of scalers.

        @remarks The list is empty if the data is not scaled.
                 Furthermore, you can use the scaler to inverse the scaled data.
                 To do so, select the scaler for the feature you want to inverse
                 e.g. scaler['c'].inverse_transform(data)
        """
        return self._scaler

    @property
    def target_scaler(self) -> StandardScaler:
        """Get the scaler for the target feature.

        @return: The scaler for the target feature.
        """
        if self._target is None:
            raise ValueError("Target is not set")
        return self._scaler[self._target]

    @property
    def time_steps_in(self) -> int:
        """Get the number of time steps for the input."""
        return self._time_steps_in

    @property
    def time_steps_out(self) -> int:
        """Get the number of time steps for the output."""
        return self._time_steps_out

    def loc_of(self, feature: str) -> int:
        """Get the location of the feature in the data.

        @param feature: The feature you want to get the location for.
        @return: The location of the feature in the data.
        """
        return self._data.columns.get_loc(feature)

    def feature_name(self, loc: int) -> str:
        """Get the name of the feature at the given location.

        @param loc: The location of the feature.
        @return: The name of the feature.
        """
        return self._data.columns[loc]

    def feature(self, feature: str) -> np.ndarray:
        """Get the feature as numpy array.

        @param feature: The feature you want to get.
        @return: The feature as numpy array.
        """
        if feature not in self._data.columns:
            raise ValueError(f"Feature {feature} not in data.")
        return self._data[feature].values

    def feature_train(self, feature: str) -> np.ndarray:
        """Get the feature of the train data as numpy array.

        @param feature: The feature you want to get.
        @return: The feature as numpy array.
        """
        if feature not in self._train_data.columns:
            raise ValueError(f"Feature {feature} not in train data.")
        return self._train_data[feature].values

    def feature_test(self, feature: str) -> np.ndarray:
        """Get the feature of the test data as numpy array.

        @param feature: The feature you want to get.
        @return: The feature as numpy array.
        """
        if feature not in self._test_data.columns:
            raise ValueError(f"Feature {feature} not in test data.")
        return self._test_data[feature].values

    def _drop_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows with nan values."""
        # Safe the header
        header = data.columns
        # Get coulums which are not numeric
        nan_columns = data.columns[data.dtypes == "object"]
        # Get rows which contain nan values
        nan_rows = data.index[data.isna().any(axis=1)]
        # Drop nan columns and rows
        data = data.drop(nan_columns, axis=1)
        data = data.drop(nan_rows, axis=0)
        # Drop the indedx column, if exists eighter named 'index' or 'Unnamed: 0'
        if "index" in data.columns:
            data = data.drop("index", axis=1)
        if "Unnamed: 0" in data.columns:
            data = data.drop("Unnamed: 0", axis=1)
        # Check if target column is still in data
        if self._target not in header and self._target is not None:
            raise ValueError(
                f"The target column {self._target} is not in the data anymore."
            )
        return data

    def _split_train_test(self, data: pd.DataFrame, test_split: float, test_length: int) -> tuple:
        """Split the data into train and test set."""

        # Adjust the length of data after extracting hat_sample
        data = data[:-self._hat_length]

        if test_split is not None:
            train_size = int(len(data) * (1 - test_split))
        else:
            train_size = len(data) - test_length

        train_data = data[:train_size]
        test_data = data[train_size:]

        # Return train_data and test_data
        return train_data, test_data

    def _scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale the data."""
        # Scale the data
        scaler = {}
        # Fit on train data and transform train and test data
        for column in data.columns:
            # Cop y the values to reshape them
            values = data[column].values
            scaler[column] = StandardScaler()
            values = values.reshape(-1, 1)
            scaler[column].fit(values)
            values = scaler[column].transform(values)
            data[column] = values
        # Save the scaler
        self._scaler = scaler
        return data
    
    def _mirror_sequence(self, sequence: pd.DataFrame) -> pd.DataFrame:
        """Mirror the sequence."""
        return sequence.iloc[::-1].reset_index(drop=True)

    def _create_samples(
        self,
        input_sequence: pd.DataFrame,
        steps_in: int,
        steps_out: int,
        overlap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create samples of x and y.

        @param input_sequence: The input sequence.
        @param steps_in: The number of time steps fed into the network.
        @param steps_out: The number of time steps the network predicts.
        @param overlap: The number of time steps that each sample will overlap.

        @return: A tuple of x and y samples.

        @remarks: The x samples are of shape (samples, time_steps, features).
                  The y samples are of shape (samples, time_steps).
                  After every sample of x there is a sample of y.
        """
        x = []
        y = []
        iterator = 0
        step_size = steps_in - overlap  # Calculate the non-overlapping step size

        if step_size < 1:
            raise ValueError("Overlap must be less than steps_in to progress the iterator.")

        while iterator + steps_in + steps_out <= len(input_sequence):
            x.append(input_sequence.iloc[iterator : iterator + steps_in].values)
            if self._target is not None:
                y.append(
                    input_sequence.iloc[iterator + steps_in : iterator + steps_in + steps_out][
                        self._target
                    ].values
                )
            iterator += step_size  # Move iterator by the non-overlapping step size

        return np.array(x), np.array(y)
