"""Testbench (unit test) for the Data_Aquirer class."""
import unittest
import argparse
import numpy as np
import tensorflow as tf
from config_tb import *
from src.composer import Composer

class Test_Composer(unittest.TestCase):
    """Test the Composer class.
    
    @remarks This is some unit test for the Composer class.
    """

    def __init__(self, methodName: str = ..., pair: str = None, fetch: bool = False, predict: bool = False, box_pts: int = 10, interval: int = None, strategy: str = 'mirrored', no_request: bool = False):
        """Initialize the testbench."""
        super().__init__(methodName)
        self.composer = Composer(pair)
        self.fetch = fetch
        self.predict = predict
        self.box_pts = box_pts
        self.interval = interval
        self.strategy = strategy
        self.no_request = no_request
        # Eleminate randomness
        np.random.seed(42)
        tf.random.set_seed(42)


    def test_composer(self):
        """Test the composer."""
        if self.fetch == False:
            from_file = True
        self.composer.summary()
        print(self.interval)
        self.composer.aquire(from_file=from_file, interval=self.interval, no_request=self.no_request)
        self.composer.calculate()
        self.composer.preprocess()
        if self.strategy == 'mirrored':
            mirrored_strategy = tf.distribute.MirroredStrategy()
            self.composer.compile(strategy=mirrored_strategy)
        else:
            self.composer.compile()
        if(self.predict == True):
            self.composer.predict(box_pts=self.box_pts)
        else:
            self.composer.fit()


def __main__(pair, fetch, predict, box_pts, interval, strategy, no_request):
    suite = unittest.TestSuite()
    suite.addTest(Test_Composer('test_composer', pair, fetch, predict, box_pts, interval, strategy, no_request))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get pair')
    parser.add_argument('--pair', type=str, help='Pair for the Composer class')
    parser.add_argument('--fetch', action='store_true', default=False, help='Fetch status for the Composer class (if False, use data from file)')
    parser.add_argument('--predict', action='store_true', default=False, help='Predict status for the Composer class (if False, fit the model))')
    parser.add_argument('--box_pts', type=int, default=0, help='Box points for the Composer class prediction (to smooth the predicted data)')
    parser.add_argument('--interval', type=int, default=None, help='Interval for the pair data (in minutes)')
    parser.add_argument('--strategy', type=str, default=False, help='Strategy to train the model')
    parser.add_argument('--no_request', action='store_true', default=False, help='No request for the Composer class (if True, use data from file)')
    args = parser.parse_args()
    __main__(args.pair, args.fetch, args.predict, args.box_pts, args.interval, args.strategy, args.no_request)