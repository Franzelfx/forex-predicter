"""Testbench for the utilizer module."""
import unittest
import traceback
from config_tb import *
from src.utilizer import Utilizer
from src.visualizer import Visualizer
from src.indicators import Indicators
from src.preprocessor import Preprocessor
from src.data_aquirer import Data_Aquirer
from src.time_series_model import TimeSeriesModel as Model

class UtilizerIntegrationTest(unittest.TestCase):
    """Test the utilizer."""

    def test_utilizer_integration(self):
        """Test the utilizer."""
        _found_start = False
        for pair in UTIL_PAIRS:
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
                # Start is today - 1 month
                data = aquirer.get(pair, MINUTES, start=UTILIZER_START_DATE, end=END, save=True)
                # Apply indicators
                indicators = Indicators(data, TEST_INDICATORS)
                data = indicators.calculate(
                    save=True, path=f"{PATH_INDICATORS}/{pair}_{MINUTES}.csv"
                )
                indicators.summary()
                # Preprocess data
                preprocessor = Preprocessor(
                    data,
                    TARGET,
                    time_steps_in=TEST_TIME_STEPS_IN,
                    time_steps_out=TEST_TIME_STEPS_OUT,
                    scale=TEST_SHIFT,
                    shift=TEST_SHIFT,
                )
                preprocessor.summary()
                # Load the model (when server is mounted)
                path = "/Volumes/lstm-server/ftp/forex-predicter/test"
                # Check, if path exists
                if not os.path.exists(path):
                    print("Path to server does not exist. Using local path.")
                    path = MODEL_PATH
                # Load the model (remove 'C:' from pair name)
                pair = pair[2:]
                model = Model(
                    path,
                    pair,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                # Directly predict from saved model
                utilizer = Utilizer(model, preprocessor)
                test_actual = utilizer.test_actual
                test_predict, y_hat = utilizer.predict(box_pts=TEST_BOX_PTS)
                visualizer = Visualizer(pair)
                path = f"{MODEL_PATH}/utilizer_test"
                visualizer.plot_prediction(path, y_hat,test_predict=test_predict, test_actual=test_actual)
            except Exception:
                traceback.print_exc()
                logging.error(traceback.format_exc())

if __name__ == "__main__":
    API_KEY = os.environ.get("API_KEY")
    unittest.main()
