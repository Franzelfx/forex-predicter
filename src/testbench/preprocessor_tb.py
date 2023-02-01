"""Testbench for the Preprocessor class."""
import unittest
import pandas as pd
from config_tb import *
import numpy as np
from matplotlib import pyplot as plt
from preprocessor import Preprocessor


class Test_Preprocessor(unittest.TestCase):
    """Test the Preprocessor class."""

    def test_x_train(self):
        """Test the x_train attribute."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        print(f"x_train shape: {preprocessor.x_train.shape}")
        # Check shape of x_train, has to be (samples, time_steps_in, features)
        self.assertEqual(preprocessor.x_train.shape[1], TEST_TIME_STEPS_IN)

    def test_y_train(self):
        """Test the y_train attribute."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            'c',
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        print(f"y_train shape: {preprocessor.y_train.shape}")
        # Check shape of y_train, has to be (samples, time_steps_out, 1)
        self.assertEqual(preprocessor.y_train.shape[1], TEST_TIME_STEPS_OUT)
    
    def test_x_test(self):
        """Test the x_test attribute."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        print(f"x_test shape: {preprocessor.x_test.shape}")
        # Check shape of x_test, has to be (samples, time_steps_in, features)
        self.assertEqual(preprocessor.x_test.shape[1], TEST_TIME_STEPS_IN)

    def test_nan_values(self):
        """Test if the data contains NaN values."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        # Convert to pandas dataframe and check for NaN values
        x_train = pd.DataFrame(preprocessor.x_train.flatten())
        y_train = pd.DataFrame(preprocessor.y_train.flatten())
        # Print where the NaN values are located
        print(x_train[x_train.isnull().any(axis=1)])
        # Print Content of the NaN values
        print(x_train[x_train.isnull().any(axis=1)].values)
        self.assertFalse(x_train.isnull().values.any())
        self.assertFalse(y_train.isnull().values.any())
    
    def test_x_y_train(self):
        """Test if the x_train and y_train data is in the correct order."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        # Check, if the first value of the second sample in x_train is the
        # same as the last value of the first sample in y_train.
        # Get x_train and y_train target values
        columns = preprocessor.data.columns
        column_loc = columns.get_loc(preprocessor.target)
        print(columns)
        print(column_loc)
        x_train_target = preprocessor.x_train[:, :, column_loc]
        y_train_target = preprocessor.y_train[:, :]
        # Scale back to original values
        x_train_target = preprocessor.scaler[preprocessor.target].inverse_transform(x_train_target)
        # Scale back to original values
        y_train_target = preprocessor.scaler[preprocessor.target].inverse_transform(y_train_target)
        # Safe x_train and y_train as csv in "prepprocessor_test" folder
        x_train_target = pd.DataFrame(x_train_target.flatten())
        y_train_target = pd.DataFrame(y_train_target.flatten())
        x_train_target.to_csv(f"{PREPROCESSOR_PATH}/x_train_target.csv")
        y_train_target.to_csv(f"{PREPROCESSOR_PATH}/y_train_target.csv")
    
if __name__ == "__main__":
    unittest.main()
