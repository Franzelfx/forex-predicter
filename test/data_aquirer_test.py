"""Testbench (unit test) for the Data_Aquirer class."""
import unittest
from config_tb import *
from src.data_aquirer import Data_Aquirer

class Test_Data_Aquirer(unittest.TestCase):
    """Test the Data_Aquirer class.
    
    @remarks This is some unit test for the Data_Aquirer class.
    """

    def __init__(self, methodName: str = ...) -> None:
        """Initialize the testbench."""
        super().__init__(methodName)
        self.aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type='full')

    def test_get_api(self):
        """Test the get method."""
        data = self.aquirer.get(PAIR, MINUTES, save=True)
        self.assertGreater(len(data), 0)
        # Check, if dataframe has the correct columns
        self.assertEqual(data.columns.tolist(), ['v', 'vw', 'o', 'c', 'h', 'l'])
    
    def test_get_file(self):
        """Test the get method."""
        data = self.aquirer.get(PAIR, MINUTES, save=True, from_file=True)
        self.assertGreater(len(data), 0)
        # Check, if dataframe has the correct columns
        self.assertEqual(data.columns.tolist(), ['v', 'vw', 'o', 'c', 'h', 'l'])

if __name__ == '__main__':
    unittest.main()