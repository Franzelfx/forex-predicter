"""Testbench for the Preprocessor class."""
import unittest
import pandas as pd
from config_tb import *
import numpy as np
from matplotlib import pyplot as plt
from src.preprocessor import Preprocessor


class Test_Preprocessor(unittest.TestCase):
    """Test the Preprocessor class."""

    def __init__(self, methodName: str = "runTest") -> None:
        """Initialize the test class."""
        super().__init__(methodName)
        test_data = pd.read_csv(PREPROCESSOR_DATA_SOURCE)
        self.preprocessor = Preprocessor(
            test_data,
            TARGET,
            test_length=TEST_LENGTH,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
        )
    
    def test_print_summary(self):
        """Print the summary of the preprocessor."""
        self.preprocessor.summary()

    def test_x_train(self):
        """Test the x_train attribute."""
        # Check shape of x_train, has to be (samples, time_steps_in, features)
        self.assertEqual(self.preprocessor.x_train.shape[1], TEST_TIME_STEPS_IN)

    def test_y_train(self):
        """Test the y_train attribute."""
        # Check shape of y_train, has to be (samples, time_steps_out)
        self.assertEqual(self.preprocessor.y_train.shape[1], TEST_TIME_STEPS_OUT)
    
    def test_x_test(self):
        """Test the x_test attribute."""
        self.assertEqual(self.preprocessor.x_test.shape[1], TEST_TIME_STEPS_IN)

    def test_y_test(self):
        """Test the y_test attribute."""
        self.assertEqual(self.preprocessor.y_test.shape[1], TEST_TIME_STEPS_OUT)

    def test_nan_values(self):
        """Test if the data contains NaN values."""
        # Convert to pandas dataframe and check for NaN values
        x_train = pd.DataFrame(self.preprocessor.x_train.flatten())
        y_train = pd.DataFrame(self.preprocessor.y_train.flatten())
        self.assertFalse(x_train.isnull().values.any())
        self.assertFalse(y_train.isnull().values.any())

    def test__train_test_set(self):
        """Plot the train and test set."""
        train_c = self.preprocessor.feature_train('c')
        test_c = self.preprocessor.feature_test('c')
        train_ma50 = self.preprocessor.feature_train('MA50')
        test_ma50 = self.preprocessor.feature_test('MA50')
        # High dpi for better quality
        plt.figure(dpi=1200)
        # Fine line width for better quality
        plt.rcParams["lines.linewidth"] = 0.25
        # Plot close price
        plt.plot(train_c, color="red", label="train_c")
        # Plot test, shiftet by the length of the train set
        plt.plot(
            np.arange(len(train_c), len(train_c) + len(test_c)),
            test_c,
            color="blue",
            label="test_c",
        )
        # Plot moving average
        plt.plot(train_ma50, color="green", label="train_ma50")
        plt.plot(
            np.arange(len(train_ma50), len(train_ma50) + len(test_ma50)),
            test_ma50,
            color="orange",
            label="test_ma50",
        )
        plt.legend()
        plt.savefig(f"{PREPROCESSOR_PATH}/train_test_set.png")

    def test_x_y_train(self):
        """Test if the x_train and y_train data is in the correct order."""
        # Check, if the first value of the second sample in x_train is the
        # same as the last value of the first sample in y_train.
        # Get x_train and y_train target values
        columns = self.preprocessor.data.columns
        column_loc = columns.get_loc(self.preprocessor.target)
        x_train_target = self.preprocessor.x_train[:, :, column_loc]
        y_train_target = self.preprocessor.y_train[:, :]
        # Save the values to csv files
        x_train_target_cs = pd.DataFrame(x_train_target.flatten())
        y_train_target_cs = pd.DataFrame(y_train_target.flatten())
        x_train_target_cs.to_csv(f"{PREPROCESSOR_PATH}/x_train_target.csv")
        y_train_target_cs.to_csv(f"{PREPROCESSOR_PATH}/y_train_target.csv")
        # Get first n_time_steps_out values of second x_train sample
        x_train_target_plt = x_train_target[1, :self.preprocessor.time_steps_out]
        # Get last n_time_steps_out values of first y_train sample
        y_train_target_plt = y_train_target[0, -self.preprocessor.time_steps_out:]
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
    
    def test___y_train_prediction(self):
        """Test if the y_train prediction is correct."""
        # Get last ample of y_train
        x_test = self.preprocessor.x_test[-1, :, self.preprocessor.loc_of(self.preprocessor.target)]
        y_test = self.preprocessor.y_test[-1, :]
        # Get prediction sample
        x_predict = self.preprocessor.x_predict[-1, :, self.preprocessor.loc_of(self.preprocessor.target)]
        # Extract the lastn_time_steps_out values of x_test
        x_test = x_test[-self.preprocessor.time_steps_out:]
        # Extract the last n_time_steps_out values of x_predict
        x_predict = x_predict[-self.preprocessor.time_steps_out:]
        # Plot the values in subplots
        fig, axs = plt.subplots(3, 1)
        # High dpi for better quality
        fig.set_dpi(300)
        axs[0].plot(x_test, color="red", label="x_test")
        # Add comment to the top of the plot
        axs[0].annotate(
            "Last x_test sample",
            xy=(0.5, 0.5),
            xytext=(0.35, 0.15),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=8,
            ha="center",
            va="center",
        )
        axs[1].plot(y_test, color="blue", label="y_test")
        # Add comment to the plot
        axs[1].annotate(
            "Last y_test sample",
            xy=(0, 0),
            xytext=(0.35, 0.15),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=8,
            ha="center",
            va="center",
        )
        axs[2].plot(x_predict, color="green", label="x_predict")
        # Add comment to the plot
        axs[2].annotate(
            "Last x_predict sample",
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
        axs[2].legend()
        plt.savefig(f"{PREPROCESSOR_PATH}/y_train_prediction.png")

    
if __name__ == "__main__":
    unittest.main()
