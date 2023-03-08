"""System test for all modules."""
import unittest
import traceback
import numpy as np
from config_tb import *
from src.model import Model
from src.utilizer import Utilizer
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
                aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
                data = aquirer.get(
                    pair, MINUTES, start=START, save=True, end=END, from_file=False
                )
                # Apply indicators
                indicators = Indicators(data, TEST_INDICATORS)
                data = indicators.calculate(
                    save=True, path=f"{PATH_INDICATORS}/{pair}_{MINUTES}.csv"
                )
                # Preprocess data
                preprocessor = Preprocessor(
                    data,
                    TARGET,
                    time_steps_in=TEST_TIME_STEPS_IN,
                    time_steps_out=TEST_TIME_STEPS_OUT,
                    test_length=TEST_TIME_STEPS_OUT,
                    scale=TEST_SCALE,
                    shift=TEST_SHIFT,
                )
                preprocessor.summary()
                model = Model(
                    MODEL_PATH,
                    pair,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                model.compile_and_fit(
                    epochs=TEST_EPOCHS,
                    hidden_neurons=TEST_NEURONS,
                    batch_size=TEST_BATCH_SIZE,
                    learning_rate=TEST_LEARNING_RATE,
                    patience=TEST_PATIENCE,
                    x_val=preprocessor.x_test,
                    y_val=preprocessor.y_test,
                )
                # Predict the next values
                utilizer = Utilizer(model, preprocessor)
                test_predict, y_hat = utilizer.predict
                # Plot the results
                visualizer = Visualizer(pair)
                path = f"{MODEL_PATH}/system_test"
                visualizer.plot_prediction(
                    path, hat=y_hat, test_actual=utilizer.test_actual
                )
            except Exception:
                traceback.print_exc()

    def moving_average(self, data, n):
        """Calculate the moving average for the given data."""
        data = np.array(data)
        ret = np.cumsum(data, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n


if __name__ == "__main__":
    unittest.main()
