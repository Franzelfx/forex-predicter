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
            scale=TEST_SCALE,
        )
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
            scale=TEST_SCALE,
        )
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
            scale=TEST_SCALE,
        )
        # Check shape of x_test, has to be (samples, time_steps_in, features)
        self.assertEqual(preprocessor.x_test.shape[1], TEST_TIME_STEPS_IN)
        # Check, if the feature values of x_test at a specific location are 
        # the same as in feature() fucntion of the Preprocessor class
        x_test = preprocessor.x_test[:, :, preprocessor.loc_of("c")]
        x_test = x_test.flatten()
        feature_c = preprocessor.feature_test("c")
        # Plot x_test and feature_x
        plt.cla()
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.plot(feature_c, label="feature_c")
        plt.plot(x_test, label="x_test")
        plt.legend()
        plt.savefig(f"{PREPROCESSOR_PATH}/x_test.png")
        # Get first time_steps_in values of feature_c
        feature_c = feature_c[:TEST_TIME_STEPS_IN]
        self.assertTrue(np.array_equal(x_test, feature_c))

    def test_nan_values(self):
        """Test if the data contains NaN values."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
        )
        # Convert to pandas dataframe and check for NaN values
        x_train = pd.DataFrame(preprocessor.x_train.flatten())
        y_train = pd.DataFrame(preprocessor.y_train.flatten())
        self.assertFalse(x_train.isnull().values.any())
        self.assertFalse(y_train.isnull().values.any())

    def test_train_test_set(self):
        """Test if the train and test set are correct."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
        )
        # Get train_data target column
        train_data = preprocessor.train_data[preprocessor.target]
        # Get test_data target column
        test_data = preprocessor.test_data[preprocessor.target]
        # Plot train_data and test_data
        x_test = preprocessor.x_test[:, :, preprocessor.loc_of("h")]
        x_test = x_test.flatten()
        # High resolution plot
        plt.figure(figsize=(20, 10))
        plt.plot(x_test, label="x_test")
        plt.plot(train_data, label="train_data")
        plt.plot(test_data, label="test_data")
        plt.legend()
        plt.savefig(f"{PREPROCESSOR_PATH}/train_test_set.png")
        # Plot h, l, c, o of x_test
        plt.figure(figsize=(20, 10))
        plt.plot(preprocessor.x_test[:, :, preprocessor.loc_of("h")].flatten(), label="h")
        plt.plot(preprocessor.x_test[:, :, preprocessor.loc_of("l")].flatten(), label="l")
        plt.plot(preprocessor.x_test[:, :, preprocessor.loc_of("c")].flatten(), label="c")
        plt.plot(preprocessor.x_test[:, :, preprocessor.loc_of("o")].flatten(), label="o")

    
    def test_x_y_train(self):
        """Test if the x_train and y_train data is in the correct order."""
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
        )
        # Check, if the first value of the second sample in x_train is the
        # same as the last value of the first sample in y_train.
        # Get x_train and y_train target values
        columns = preprocessor.data.columns
        column_loc = columns.get_loc(preprocessor.target)
        x_train_target = preprocessor.x_train[:, :, column_loc]
        y_train_target = preprocessor.y_train[:, :]
        # Scale back to original values
        x_train_target = preprocessor.scaler[preprocessor.target].inverse_transform(x_train_target)
        # Scale back to original values
        y_train_target = preprocessor.scaler[preprocessor.target].inverse_transform(y_train_target)
        # Safe x_train and y_train as csv in "prepprocessor_test" folder
        x_train_target_cs = pd.DataFrame(x_train_target.flatten())
        y_train_target_cs = pd.DataFrame(y_train_target.flatten())
        x_train_target_cs.to_csv(f"{PREPROCESSOR_PATH}/x_train_target.csv")
        y_train_target_cs.to_csv(f"{PREPROCESSOR_PATH}/y_train_target.csv")
        # Get first n_time_steps_out values of second x_train sample
        x_train_target_plt = x_train_target[1, :preprocessor.time_steps_out]
        # Get last n_time_steps_out values of first y_train sample
        y_train_target_plt = y_train_target[0, -preprocessor.time_steps_out:]
        # Check if the values are the same
        self.assertTrue(np.array_equal(x_train_target_plt, y_train_target_plt))
        # Plot the values in subplots
        fig, axs = plt.subplots(2, 1)
        # High dpi for better quality
        fig.set_dpi(300)
        axs[0].plot(x_train_target_plt, color="red", label="x_train_target")
        # Add comment to the top of the plot
        axs[0].annotate(
            "First n_time_steps_out values of second x_train sample",
            xy=(0.5, 0.5),
            xytext=(0.35, 0.15),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=8,
            ha="center",
            va="center",
        )
        axs[1].plot(y_train_target_plt, color="blue", label="y_train_target")
        # Add comment to the plot
        axs[1].annotate(
            "Last n_time_steps_out values of first y_train sample",
            xy=(0, 0),
            xytext=(0.35, 0.15),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=8,
            ha="center",
            va="center",
        )
        axs[0].legend()
        axs[1].legend()
        plt.savefig(f"{PREPROCESSOR_PATH}/x_y_train.png")


    
if __name__ == "__main__":
    unittest.main()
