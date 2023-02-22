"""Testbench for the utilizer module."""
import unittest
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
        for pair in REQUEST_PAIRS:
            # Get data from the API
            aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type='full')
            data = aquirer.get(pair, MINUTES, save=True)
            # Apply indicators
            indicators = Indicators(data, TEST_INDICATORS)
            data = indicators.calculate(save=True, path=f"{PATH_INDICATORS}/{pair}_{MINUTES}.csv")
            # Preprocess data
            preprocessor = Preprocessor(
                data,
                TARGET,
                time_steps_in=TEST_TIME_STEPS_IN,
                time_steps_out=TEST_TIME_STEPS_OUT,
                scale=TEST_SCALE,
            )
            preprocessor.summary()
            # Load the model
            model = Model(
                MODEL_PATH,
                pair,
                preprocessor.x_train,
                preprocessor.y_train,
            )
            # Last known value
            last_known = preprocessor.last_known_value
            # Directly predict from saved model
            utilizer = Utilizer(model, preprocessor.x_test)
            prediction = utilizer.predict(TEST_TIME_STEPS_OUT, preprocessor.target_scaler, ma_period=50, last_known=last_known)
            visualizer = Visualizer(pair)
            path = f"{MODEL_PATH}/utilizer_test"
            visualizer.plot_prediction(prediction, path)

if __name__ == "__main__":
    unittest.main()