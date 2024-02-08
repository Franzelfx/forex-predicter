"""Testbench (unit test) for the Data_Aquirer class."""
import unittest
import argparse
from config_tb import *
import tensorflow as tf

class Test_Composer(unittest.TestCase):
    """Test the Composer class.
    
    @remarks This is some unit test for the Composer class.
    """

    def __init__(self, methodName: str = ..., pair: str = None, fetch: bool = False, predict: bool = False, box_pts: int = 10, interval: int = None, strategy: str = 'mirrored', no_request: bool = False, end_time: str = None, test: bool = False, continue_training: bool = False):
        """Initialize the testbench."""
        from src.composer import Composer
        super().__init__(methodName)
        self.composer = Composer(pair)
        self.fetch = fetch
        self.predict = predict
        self.box_pts = box_pts
        self.interval = interval
        self.strategy = strategy
        self.no_request = no_request
        self.end_time = end_time
        self.test = test
        self.continue_training = continue_training

    def test_composer(self):
        # Eleminate randomness
        if self.fetch == False:
            from_file = True
        self.composer.summary()
        print(self.interval)
        # If we want to predict, we don't care about the beginning of the data
        ignore_start = self.predict
        self.composer.aquire(from_file=from_file, interval=self.interval, no_request=self.no_request, ignore_start=ignore_start, end_time=self.end_time)
        self.composer.calculate()
        self.composer.preprocess()
        if self.strategy == 'mirrored':
            mirrored_strategy = tf.distribute.MirroredStrategy()
            self.composer.compile(strategy=mirrored_strategy)
        else:
            self.composer.compile()
        if(self.predict == True):
            self.composer.predict(box_pts=self.box_pts, test=self.test)
            self.composer.dump()
        else:
            self.composer.fit(continue_training=self.continue_training)


def __main__(pair, fetch, predict, box_pts, interval, strategy, no_request, end_time, test, continue_training):
    suite = unittest.TestSuite()
    suite.addTest(Test_Composer('test_composer', pair, fetch, predict, box_pts, interval, strategy, no_request, end_time, test, continue_training))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get pair')
    parser.add_argument('--pair', type=str, help='Pair for the Composer class')
    parser.add_argument('--fetch', action='store_true', default=False, help='Fetch status for the Composer class (if False, use data from file)')
    parser.add_argument('--predict', action='store_true', default=False, help='Predict status for the Composer class (if False, fit the model))')
    parser.add_argument('--box-pts', type=int, default=0, help='Box points for the Composer class prediction (to smooth the predicted data)')
    parser.add_argument('--interval', type=int, default=None, help='Interval for the pair data (in minutes)')
    parser.add_argument('--strategy', type=str, default=False, help='Strategy to train the model')
    parser.add_argument('--no-request', action='store_true', default=False, help='No request for the Composer class (if True, use data from file)')
    parser.add_argument('--test', action='store_true', default=False, help='Run with test data')
    parser.add_argument('--end', type=str, default=None, help='End time for data acquisition in yyyy-mm-dd format')
    parser.add_argument('--gpu', type=str, default=None, help='GPU to use for training')
    parser.add_argument('--continue-training' , action='store_true', default=True, help='Continou training of the model')
    args = parser.parse_args()

    if args.end is None:
        args.end = date.today().strftime("%Y-%m-%d")

    if args.gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[int(args.gpu)], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[int(args.gpu)], True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)

    # If no pair is specified, iterate over all JSON files in src/recipes
    if args.pair is None:
        recipe_files = os.listdir('../src/recipes')

        for recipe_file in recipe_files:
            try:
                if recipe_file.endswith('_recipe.json'):
                    pair = recipe_file.replace('_recipe.json', '')

                    if pair not in ['BTCUSD', 'ETHUSD']:
                        prefix = 'C:' if pair[0] != 'X' else 'X:'
                        pair = prefix + pair
                    
                    __main__(pair, args.fetch, args.predict, args.box_pts, args.interval, args.strategy, args.no_request, args.end, args.test, args.continue_training)
            except Exception as e:
                print(f'Error while processing {recipe_file}: {e}')
                continue
    else:
        __main__(args.pair, args.fetch, args.predict, args.box_pts, args.interval, args.strategy, args.no_request, args.end, args.test, args.continue_training)