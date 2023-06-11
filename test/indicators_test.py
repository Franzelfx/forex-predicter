"""Testbench (unit test) for the Indicators class."""
import unittest
import pandas as pd
from config_tb import *
from pandas import DataFrame
import matplotlib.pyplot as plt
from src.indicators import Indicators

class Test_Indicators(unittest.TestCase):
    """Test the Indicators class."""

    def __init__(self, methodName: str = "runTest") -> None:
        """Initialize the test class."""
        super().__init__(methodName)
        # Get test data
        self.test_data = pd.read_csv(INDICATORS_DATA_SOURCE)
        # Create an instance of the Indicators class
        self.indicators = Indicators(PATH_INDICATORS, PAIR, self.test_data, TEST_INDICATORS)
        self.data: DataFrame = self.indicators.calculate(save=True)

    def _summary(self):
        """Test the summary method."""
        self.indicators.summary()

    def test_calculate_indicators(self):
        """Test the calculate_indicators method."""
        self._summary()
        self.assertGreater(len(self.data), 0)
        # Check, if dataframe has the colums from available indicators
        self.assertTrue(self._check_nan_values(self.data))
        self.assertTrue(self._check_presence(self.data, EXPECTED_COLUMNS))
        self.assertTrue(self._chek_column_len(self.data, EXPECTED_COLUMNS))
        self._plot_indicators_MA50()
        self._plot_indicators_MA200()
    
    def _check_presence(self, data: pd.DataFrame, indicators: list) -> bool:
        """Check, if the indicators are in the dataframe.

        @param data: Dataframe with the data.
        @param indicators: List of indicators.
        @return: True, if all indicators are in the dataframe.
        """
        for indicator in indicators:
            if indicator not in data.columns:
                return False
        return True
    
    def _chek_column_len(self, data: pd.DataFrame, indicators: list) -> bool:
        """Check, if the indicators in the dataframe have the same length.

        @param data: Dataframe with the data.
        @param indicators: List of indicators.
        @return: True, if all indicators are in the dataframe.
        """
        for indicator in indicators:
            if len(data[indicator]) != len(data):
                return False
        return True

    def _check_nan_values(self, data: pd.DataFrame) -> bool:
        """Check, if there are some NaN values in the dataframe.

        @param data: Dataframe with the data.
        @return: True, if there are some NaN values.
        """
        if data.isnull().values.any():
            return False
        return True
    
    def _plot_indicators_MA50(self):
        """Plot close and MA50."""
        # Plot the MA50 and MA200
        plt.cla()
        plt.clf()
        plt.figure(dpi=1200)
        plt.rcParams["lines.linewidth"] = 0.25
        plt.plot(self.data['c'], label="close")
        plt.plot(self.data["MA50"], label="MA50")
        plt.legend(loc="upper left")
        plt.savefig(f"{PATH_INDICATORS}/{PAIR}_{MINUTES_TRAIN}_MA50.png")
    
    def _plot_indicators_MA200(self):
        """Test the plot_indicators method."""
        # Plot closa and MA200
        plt.cla()
        plt.clf()
        plt.figure(dpi=1200)
        plt.rcParams["lines.linewidth"] = 0.25
        plt.plot(self.data['c'], label="close")
        plt.plot(self.data["MA200"], label="MA200")
        plt.legend(loc="upper left")
        plt.savefig(f"{PATH_INDICATORS}/{PAIR}_{MINUTES_TRAIN}_MA200.png")

if __name__ == '__main__':
    unittest.main()