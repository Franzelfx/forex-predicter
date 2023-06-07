"""Testbench for the model class."""
import unittest
import tensorflow as tf
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
        from_saved_file = os.getenv("FROM_SAVED_FILE")
        use_multiple_gpus = os.environ.get("USE_MULTIPLE_GPUS")
        print(use_multiple_gpus)
        print(True)
        # Data
        aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
        api_data = aquirer.get(
            PAIR, MINUTES, start=START, end=END, save=True, from_file=from_saved_file
        )
        # Indicators
        indicators = Indicators(PATH_INDICATORS, PAIR, api_data, TEST_INDICATORS)
        indicator_data = indicators.calculate(save=True)
        preprocessor = Preprocessor(
            indicator_data,
            TARGET,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
            shift=TEST_SHIFT,
        )
        preprocessor.summary()
        # Model
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Run for testing
        if use_multiple_gpus:
            strategy = tf.distribute.MirroredStrategy()
        model.compile(
            learning_rate=TEST_LEARNING_RATE,
            hidden_neurons=TEST_NEURONS,
            strategy=strategy if use_multiple_gpus else None,
        )
        model.fit(
            epochs=TEST_EPOCHS,
            batch_size=TEST_BATCH_SIZE,
            validation_split=TEST_VALIDATION_SPLIT,
            patience=TEST_PATIENCE,
        )

if __name__ == "__main__":
    # get API_KEY from environment variable
    API_KEY = os.environ.get("API_KEY")
    unittest.main()
