"""Testbench (unit test) for the Indicators class."""
import unittest
import pandas as pd
from config_tb import *
from indicators import Indicators


class Test_Indicators(unittest.TestCase):
    """Test the Indicators class.

    @remarks This is some integration test for the Indicators class.
                Where the Data_Aquirer class is used to get the data.
    """

    def test_calculate_indicators(self):
        """Test the calculate_indicators method."""
        # Get some test data
        test_data = pd.read_csv(TEST_DATA_SOURCE)
        indicators = Indicators(test_data, TEST_INDICATORS)
        data = indicators.calculate_indicators()
        self.assertGreater(len(data), 0)
        # Check, if dataframe has the correct columns
        self.assertEqual(
            data.columns.tolist(),
            [
                "v",
                "vw",
                "o",
                "c",
                "h",
                "l",
                "n",
                "ATR",
                "BOLLINGER",
                "MA50",
                "MA200",
                "MACD",
                "RSI",
                "STOCHASTIC",
            ],
        )
