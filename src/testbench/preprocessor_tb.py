"""Testbench for the Preprocessor class."""
import unittest
import pandas as pd
from config_tb import *
from preprocessor import Preprocessor


class Test_Preprocessor(unittest.TestCase):
    """Test the Preprocessor class."""

    def test_intersection(self):
        """Test the preprocess method."""
        # Data in the samples intersect by intersection_factor * time_steps
        # check, if the data has the desired intersection. If so, the
        # the number of samples should increase when intersection_factor
        # increases.
        first_run = True
        test_data = pd.read_csv(TEST_DATA_SOURCE)
        intersection_factor = 0.1
        while intersection_factor < 0.8:
            intersection_factor += 0.01
            preprocessor = Preprocessor(
                test_data,
                "c",
                test_split=TEST_SPLIT,
                time_steps_in=TEST_TIME_STEPS_IN,
                time_steps_out=TEST_TIME_STEPS_OUT,
                intersection_factor=intersection_factor,
                scale=TEST_SCALE,
            )
            if not first_run:
                self.assertGreaterEqual(
                    preprocessor.x_train.shape[0], previous_x_train.shape[0]
                )
            previous_x_train = preprocessor.x_train
            first_run = False

    def test_x_train(self):
        """Test the x_train attribute."""
        test_data = pd.read_csv(TEST_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        # Check shape of x_train, has to be (samples, time_steps_in, 7)
        self.assertEqual(preprocessor.x_train.shape[1], TEST_TIME_STEPS_IN)
        self.assertEqual(preprocessor.x_train.shape[2], 7)

    def test_y_train(self):
        """Test the y_train attribute."""
        test_data = pd.read_csv(TEST_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            'c',
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        # Check shape of y_train, has to be (samples, time_steps_out, 1)
        self.assertEqual(preprocessor.y_train.shape[1], TEST_TIME_STEPS_OUT)
        self.assertEqual(preprocessor.y_train.shape[2], 1)


if __name__ == "__main__":
    unittest.main()
