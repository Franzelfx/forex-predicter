"""Testbench for the utilizer module."""
import unittest
import numpy as np
import pandas as pd
from config_tb import *
from src.model import Model
from src.utilizer import Utilizer
from src.visualizer import Visualizer
from src.preprocessor import Preprocessor

class UtilizerTest(unittest.TestCase):
    """Test the utilizer."""

    def test_utilizer(self):
        """Test the utilizer."""
        test_data = pd.read_csv(MODEL_DATA_SOURCE)
        # Preprocess the data
        preprocessor = Preprocessor(
            test_data,
            TARGET,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
        )
        preprocessor.summary()
        # Load the model
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Last known value
        last_known_x = preprocessor.last_known_x
        last_known_y = preprocessor.last_known_y
        x_predict_cut = preprocessor.data[-(preprocessor.time_steps_in+preprocessor.time_steps_out):-preprocessor.time_steps_out].values
        # Add samples dimension
        x_predict_cut = np.expand_dims(x_predict_cut, axis=0)
        # Directly predict from saved model
        utilizer_test = Utilizer(model, preprocessor.x_test)
        utilizer_hat = Utilizer(model, preprocessor.x_predict)
        utilizer_cut = Utilizer(model, x_predict_cut)
        utilizer_prediction_Set = Utilizer(model, preprocessor.prediction_set)
        prediction_test = utilizer_test.predict(TEST_TIME_STEPS_OUT, scaler=preprocessor.target_scaler, ma_period=50, last_known=last_known_x)
        prediction_hat = utilizer_hat.predict(TEST_TIME_STEPS_OUT, scaler=preprocessor.target_scaler, ma_period=50, last_known=last_known_y)
        prediction_cut = utilizer_cut.predict(TEST_TIME_STEPS_OUT, scaler=preprocessor.target_scaler, ma_period=50, last_known=last_known_x)
        prediction_set = utilizer_prediction_Set.predict(TEST_TIME_STEPS_OUT, scaler=preprocessor.target_scaler, ma_period=50, last_known=last_known_x)
        # Get only last time steps out from prediction set
        prediction_set = prediction_set[-preprocessor.time_steps_out:]
        visualizer = Visualizer(PAIR)
        path = f"{MODEL_PATH}/utilizer_test"
        visualizer.plot_prediction(prediction_test, path, extra_info=f"test")
        visualizer.plot_prediction(prediction_hat, path, extra_info=f"hat")
        visualizer.plot_prediction(prediction_cut, path, extra_info=f"cut")
        visualizer.plot_prediction(prediction_set, path, extra_info=f"set")
        # Check, if prediction_test and prediction_hat are not the same
        self.assertNotEqual(prediction_test, prediction_hat)

if __name__ == "__main__":
    unittest.main()