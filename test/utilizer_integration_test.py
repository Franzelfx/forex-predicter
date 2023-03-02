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
        for pair in UTIL_PAIRS:
            try:
                # Get data from the API
                aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
                # Start is today - 1 month
                data = aquirer.get(pair, MINUTES, end=END, save=True, from_file=True)
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
                path = "/Volumes/lstm-server/ftp/forex-predicter/test/checkpoints"
                # Check, if path exists
                if not os.path.exists(path):
                    path = MODEL_PATH
                # Load the model
                model = Model(
                    MODEL_PATH,
                    pair,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                # Directly predict from saved model
                utilizer = Utilizer(model, preprocessor)
                test_actual = utilizer.test_actual
                test_predict, y_hat = utilizer.predict
                visualizer = Visualizer(pair)
                path = f"{MODEL_PATH}/utilizer_test"
                # visualizer.plot_prediction(prediction_train, path, extra_info=f"train")
                visualizer.plot_prediction(path, hat=y_hat, test_actual=test_actual, test_predict=test_predict)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    unittest.main()
