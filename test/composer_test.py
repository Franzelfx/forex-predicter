"""Testbench (unit test) for the Data_Aquirer class."""
import unittest
import argparse
from config_tb import *
from src.composer import Composer

class Test_Composer(unittest.TestCase):
    """Test the Composer class.
    
    @remarks This is some unit test for the Composer class.
    """

    def __init__(self, methodName: str = ..., pair: str = None, fetch=False) -> None:
        """Initialize the testbench."""
        super().__init__(methodName)
        self.composer = Composer(pair)
        self.fetch = fetch

    def test_composer(self):
        """Test the composer."""
        if self.fetch == False:
            from_file = True
        self.composer.summary()
        self.composer.aquire(from_file=from_file)
        self.composer.calculate()
        self.composer.preprocess()
        self.composer.compile()
        self.composer.fit()


def __main__(pair):
    suite = unittest.TestSuite()
    suite.addTest(Test_Composer('test_composer', pair))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get pair')
    parser.add_argument('--pair', type=str, help='Pair for the Composer class')
    parser.add_argument('--fetch', type=lambda x: (str(x).lower() == 'true'), default=False, help='Fetch status for the Composer class')
    args = parser.parse_args()
    __main__(args.pair, args.fetch)