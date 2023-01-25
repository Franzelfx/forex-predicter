"""Testbench (unit test) for the Data_Aquirer class."""
import unittest
from config_tb import *
from data_aquirer import Data_Aquirer

class Test_Data_Aquirer(unittest.TestCase):
    """Test the Data_Aquirer class.
    
    @remarks This is some unit test for the Data_Aquirer class.
    """

    def test_get_api(self):
        """Test the get method."""
        aquirer = Data_Aquirer(PATH, API_KEY, TIME_FORMAT)
        data = aquirer.get(PAIR, MINUTES, DATE_START, DATE_END, save=True)
        self.assertGreater(len(data), 0)
        # Check, if dataframe has the correct columns
        self.assertEqual(data.columns.tolist(), ['v', 'vw', 'o', 'c', 'h', 'l', 'n'])
    
    def test_get_file(self):
        """Test the get method."""
        aquirer = Data_Aquirer(PATH, API_KEY, TIME_FORMAT)
        data = aquirer.get(PAIR, MINUTES, DATE_START, DATE_END, save=True, from_file=True)
        self.assertGreater(len(data), 0)
        # Check, if dataframe has the correct columns
        self.assertEqual(data.columns.tolist(), ['v', 'vw', 'o', 'c', 'h', 'l', 'n'])

if __name__ == '__main__':
    unittest.main()