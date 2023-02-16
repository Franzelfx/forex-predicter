"""Testbench for the model class."""
import unittest
import pandas as pd
from config_tb import *
from src.model import Model
from src.visualizer import Visualizer
from src.preprocessor import Preprocessor

class Test_Model(unittest.TestCase):
    """Integration test for the Model class.

    @remarks This test is not a unit test, but an integration test. It tests the
                preprocessor and model class together.
    """

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        test_data = pd.read_csv(MODEL_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            TARGET,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
        )
        preprocessor.summary()
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Run for testing
        model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE, learning_rate=TEST_LEARNING_RATE)
        # Predict the next values
        x_test = preprocessor.x_test
        prediction = model.predict(x_test, scaler=preprocessor.target_scaler)
        # Reduce to time_steps_out
        prediction = prediction[:TEST_TIME_STEPS_OUT]
        y_test = preprocessor.y_test[:TEST_TIME_STEPS_OUT]
        # Plot the results
        visualizer = Visualizer(PAIR)
        path = f"{MODEL_PATH}/model_test/{MODEL_NAME}"
        visualizer.plot_prediction(prediction, y_test, path)

if __name__ == "__main__":
    unittest.main()
