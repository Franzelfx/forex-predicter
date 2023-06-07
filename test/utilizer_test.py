"""Testbench for the utilizer module."""
import unittest
import traceback
from config_tb import *
from src.model import Model
from src.utilizer import Utilizer
from src.visualizer import Visualizer
from src.indicators import Indicators
from src.preprocessor import Preprocessor
from src.data_aquirer import Data_Aquirer


class UtilizerIntegrationTest(unittest.TestCase):
    """Test the utilizer."""

    def test_utilizer_integration(self):
        """Test the utilizer."""
        _found_start = False
        start_pair = os.environ.get("START_PAIR")
        use_data_from_file = os.environ.get("FROM_SAVED_FILE")
        for pair in UTIL_PAIRS:
            # Try to get environment variables
            try:
                # If START_PAIR is set, skip all previous pairs
                if start_pair and _found_start is False:
                    if pair == os.environ.get("START_PAIR"):
                        _found_start = True
                        print(f"Starting with pair: {pair}")
                    else:
                        continue
                # Get data
                aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
                api_data = aquirer.get(
                    pair,
                    MINUTES,
                    start=UTILIZER_START_DATE,
                    end=END,
                    save=True,
                    from_file=use_data_from_file,
                )
                # Apply indicators
                indicators = Indicators(PATH_INDICATORS, pair, api_data, TEST_INDICATORS)
                indicator_data = indicators.calculate(save=True)
                indicators.summary()
                # Preprocess data
                preprocessor = Preprocessor(
                    indicator_data,
                    TARGET,
                    time_steps_in=TEST_TIME_STEPS_IN,
                    time_steps_out=TEST_TIME_STEPS_OUT,
                    scale=TEST_SHIFT,
                    shift=TEST_SHIFT,
                )
                preprocessor.summary()
                # Load model
                model = Model(
                    MODEL_PATH,
                    pair,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                # Directly predict from saved model
                utilizer = Utilizer(model, preprocessor)
                test_actual = utilizer.test_actual
                test_predict, y_hat = utilizer.predict(box_pts=TEST_BOX_PTS)
                # Visualize prediction
                visualizer = Visualizer(pair)
                path = f"{MODEL_PATH}/utilizer_test"
                visualizer.plot_prediction(
                    path, y_hat, test_predict=test_predict, test_actual=test_actual
                )
            except Exception:
                traceback.print_exc()
                logging.error(traceback.format_exc())


if __name__ == "__main__":
    API_KEY = os.environ.get("API_KEY")
    unittest.main()
