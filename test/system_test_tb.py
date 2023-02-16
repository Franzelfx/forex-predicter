"""System test for all modules."""
import unittest
import numpy as np
import pandas as pd
from config_tb import *
from src.model import Model
from src.indicators import Indicators
from src.visualizer import Visualizer
from src.data_aquirer import Data_Aquirer
from src.preprocessor import Preprocessor

class SystemTest(unittest.TestCase):
    """Test the system."""

    def test_system(self):
        """Test the system."""
        for pair in REQUEST_PAIRS:
            try:
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
                model = Model(
                    MODEL_PATH,
                    pair,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE, learning_rate=TEST_LEARNING_RATE)
                # Predict the next values
                x_test = preprocessor.x_test
                prediction = model.predict(x_test, scaler=preprocessor.target_scaler)
                # Reduce to time_steps_out
                prediction = prediction[:TEST_TIME_STEPS_OUT]
                y_test = preprocessor.y_test[:TEST_TIME_STEPS_OUT]
                # Plot the results
                visualizer = Visualizer(PAIR)
                visualizer.plot_prediction(prediction, y_test, MODEL_NAME)
            except Exception:
                continue

    def moving_average(self, data, n):
        """Calculate the moving average for the given data."""
        data = np.array(data)
        ret = np.cumsum(data, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


if __name__ == "__main__":
    unittest.main()