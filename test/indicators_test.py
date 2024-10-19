import unittest
import pandas as pd
from unittest.mock import patch
from src.indicators import Indicators
import os

class TestIndicatorsParallelCalculation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the real data from the specified path
        data_path = '/home/fabian/forex-predicter/src/pairs/AUDCHF_5.csv'
        if os.path.exists(data_path):
            cls.sample_data = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"CSV file not found at {data_path}")

        # Requested indicators including new ones
        cls.requested_indicators = [
            'ATR', 'BOLLINGER', 'MA5', 'MA25', 'MA50', 'MA200', 'MACD', 
            'OBV', 'RSI', 'STOCHASTIC', 'VoRSI', 'HT_TRENDLINE', 'HT_TRENDMODE', 
            'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'MFI', 
            'MOM', 'PLUS_DI', 'PLUS_DM'
        ]

    def test_calculate_indicators_in_parallel(self):
        # Initialize Indicators object with real data
        indicators = Indicators(
            path="dummy_path", 
            pair="AUDCHF", 
            data=self.sample_data.copy(), 
            requested=self.requested_indicators
        )
        
        # Call the method to calculate in parallel
        result_data = indicators.calculate_indicators_in_parallel()
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result_data, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
