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
        for pair in REQUEST_PAIRS:
            try:
                # Get data from the API
                aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type='full')
                #TODO: Fix the from_file=True
                data = aquirer.get(pair, MINUTES, save=True, from_file=True)
                # Apply indicators
                indicators = Indicators(data, TEST_INDICATORS)
                data = indicators.calculate(save=True, path=f"{PATH_INDICATORS}/{pair}_{MINUTES}.csv")
                indicators.summary()
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
                last_known_x = preprocessor.last_known_x
                last_known_y = preprocessor.last_known_y
                # Directly predict from saved model
                utilizer_test = Utilizer(model, preprocessor.x_test)
                utilizer_hat = Utilizer(model, preprocessor.x_predict)
                # TODO: Check, why the scaling is not working
                prediction_test = utilizer_test.predict(TEST_TIME_STEPS_OUT, scaler=preprocessor.target_scaler, ma_period=50, last_known=last_known_y)
                prediction_hat = utilizer_hat.predict(TEST_TIME_STEPS_OUT, scaler=preprocessor.target_scaler, ma_period=50, last_known=last_known_x)
                # Scale the prediction
                if TEST_SCALE:
                    scaler = preprocessor.target_scaler
                    prediction_test = scaler.inverse_transform(prediction_test.reshape(-1, 1)).flatten()
                    prediction_hat = scaler.inverse_transform(prediction_hat.reshape(-1, 1)).flatten()
                visualizer = Visualizer(pair)
                path = f"{MODEL_PATH}/utilizer_test"
                visualizer.plot_prediction(prediction_test, path, extra_info=f"test")
                visualizer.plot_prediction(prediction_hat, path, extra_info=f"hat")
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    unittest.main()