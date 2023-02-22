"""Testbench for the utilizer module."""
import unittest
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
        last_known = preprocessor.last_known_value
        # Directly predict from saved model
        utilizer = Utilizer(model, preprocessor.x_test)
        prediction = utilizer.predict(TEST_TIME_STEPS_OUT, preprocessor.target_scaler, ma_period=50, last_known=last_known)
        visualizer = Visualizer(PAIR)
        path = f"{MODEL_PATH}/utilizer_test"
        visualizer.plot_prediction(prediction, path)

if __name__ == "__main__":
    unittest.main()