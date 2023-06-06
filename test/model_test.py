"""Testbench for the model class."""
import unittest
import pandas as pd
from config_tb import *
from src.model import Model
from src.visualizer import Visualizer
from src.indicators import Indicators
from src.preprocessor import Preprocessor
from src.data_aquirer import Data_Aquirer


class Test_Model(unittest.TestCase):
    """Integration test for the Model class.

    @remarks This test is not a unit test, but an integration test. It tests the
                preprocessor and model class together.
    """

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
        from_saved_file = os.getenv("FROM_SAVED_FILE")
        test_data = aquirer.get(
            PAIR, MINUTES, start=START, end=END, save=True, from_file=from_saved_file
        )
        # Indicators
        indicators = Indicators(test_data, TEST_INDICATORS)
        test_data = indicators.calculate(save=True)
        preprocessor = Preprocessor(
            test_data,
            TARGET,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
            shift=TEST_SHIFT,
        )
        preprocessor.summary()
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Run for testing
        model.compile_and_fit(
            epochs=TEST_EPOCHS,
            patience=TEST_PATIENCE,
            batch_size=TEST_BATCH_SIZE,
            hidden_neurons=TEST_NEURONS,
            learning_rate=TEST_LEARNING_RATE,
            branched_model=TEST_BRANCHED_MODEL,
            validation_split=TEST_VALIDATION_SPLIT,
        )


if __name__ == "__main__":
    # get API_KEY from environment variable
    API_KEY = os.environ.get("API_KEY")
    unittest.main()
