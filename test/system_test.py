"""System test for machine learning training."""

import os
import unittest
import traceback
from config_tb import *
from src.model import Model
from src.utilizer import Utilizer
from src.indicators import Indicators
from src.visualizer import Visualizer
from src.data_aquirer import Data_Aquirer
from src.preprocessor import Preprocessor

class SystemTest(unittest.TestCase):
    """Test the system."""

    def test_system(self):
        """Test the system."""
        _found_start = False
        for pair in REQUEST_PAIRS:
            try:
                # If START_PAIR is set, skip all previous pairs
                if os.environ.get("START_PAIR") and _found_start is False:
                    if pair == os.environ.get("START_PAIR"):
                        _found_start = True
                        print(f"Starting with pair: {pair}")
                    else:
                        continue
                # Get data from the API
                aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
                data = aquirer.get(
                    pair, MINUTES_TRAIN, start=START, save=True, end=END, from_file=False
                )
                # Apply indicators
                indicators = Indicators(PATH_INDICATORS, pair, data, TEST_INDICATORS)
                data = indicators.calculate(save=True)
                # Preprocess data
                preprocessor = Preprocessor(
                    data,
                    TARGET,
                    time_steps_in=TEST_TIME_STEPS_IN,
                    time_steps_out=TEST_TIME_STEPS_OUT,
                    test_length=TEST_TIME_STEPS_OUT,
                    scale=TEST_SCALE,
                    shift=TEST_SHIFT,
                )
                preprocessor.summary()
                model = Model(
                    MODEL_PATH,
                    pair,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                model.compile_and_fit(
                    epochs=TEST_EPOCHS,
                    hidden_neurons=TEST_NEURONS,
                    batch_size=TEST_BATCH_SIZE,
                    learning_rate=TEST_LEARNING_RATE,
                    patience=TEST_PATIENCE,
                    validation_split=TEST_VALIDATION_SPLIT,
                )
                # Predict the next values
                utilizer = Utilizer(model, preprocessor)
                test_actual = utilizer.test_actual
                test_predict, y_hat = utilizer.predict(box_pts=TEST_BOX_PTS)
                # Plot the results
                visualizer = Visualizer(pair)
                path = f"{MODEL_PATH}/system_test"
                visualizer.plot_prediction(path, y_hat,test_predict=test_predict, test_actual=test_actual)
            except Exception:
                traceback.print_exc()
                logging.error(traceback.format_exc())
    
if __name__ == "__main__":
    # get API_KEY from environment variable
    API_KEY = os.environ.get("API_KEY")
    unittest.main()