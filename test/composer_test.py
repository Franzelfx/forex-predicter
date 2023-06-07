"""Testbench (unit test) for the Data_Aquirer class."""
import unittest
from config_tb import *
from src.composer import Composer

class Test_Composer(unittest.TestCase):
    """Test the Composer class.
    
    @remarks This is some unit test for the Composer class.
    """

    def __init__(self, methodName: str = ...) -> None:
        """Initialize the testbench."""
        super().__init__(methodName)
        self.composer = Composer(PAIR)

    def test_composer(self):
        """Test the composer."""
        self.composer.summary()
        #self.composer.compose()

if __name__ == '__main__':
    unittest.main()