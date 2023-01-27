"""Testbench (unit test) for the Indicators class."""
import unittest
import pandas as pd
from config_tb import *
from indicators import Indicators

class Test_Indicators(unittest.TestCase):
    """Test the Indicators class."""

    def test_calculate_indicators(self):
        """Test the calculate_indicators method."""
        # Get some test data
        test_data = pd.read_csv(TEST_DATA_SOURCE)
        indicators = Indicators(test_data, TEST_INDICATORS)
        data = indicators.calculate_indicators()
        self.assertGreater(len(data), 0)
        # Check, if dataframe has the colums from available indicators
        available = indicators.available
        self.assertTrue(self._check_presence(data, EXPECTED_COLUMNS))
        self.assertTrue(self._chek_column_len(data, EXPECTED_COLUMNS))
    
    def _check_presence(self, data: pd.DataFrame, indicators: list) -> bool:
        """Check, if the indicators are in the dataframe.

        @param data: Dataframe with the data.
        @param indicators: List of indicators.
        @return: True, if all indicators are in the dataframe.
        """
        for indicator in indicators:
            if indicator not in data.columns:
                return False
        print(f"Present columns: {data.columns.tolist()}")
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
            print(f"Lenght of {indicator}: {len(data[indicator])}")
        return True

if __name__ == '__main__':
    unittest.main()