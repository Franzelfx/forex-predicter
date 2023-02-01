"""System test for all modules."""
import unittest
from config_tb import *
from model import Model
from indicators import Indicators
from data_aquirer import Data_Aquirer
from preprocessor import Preprocessor
from matplotlib import pyplot as plt

class SystemTest(unittest.TestCase):
    """Test the system."""

    def test_system(self):
        """Test the system."""
        request_pairs = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
        # Get data
        for pair in request_pairs:
            data_aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, TIME_FORMAT)
            data = data_aquirer.get(PAIR, MINUTES, DATE_START, DATE_END, save=True)
            # Create indicators
            indicators = Indicators(data, TEST_INDICATORS)
            data = indicators.calculate(save=True, path=PATH_INDICATORS)
            # Preprocess data
            preprocessor = Preprocessor(
                data,
                "c",
                test_split=TEST_SPLIT,
                time_steps_in=TEST_TIME_STEPS_IN,
                time_steps_out=TEST_TIME_STEPS_OUT,
                intersection_factor=0.5,
                scale=True,
            )
            # Create and train model
            model = Model(
                MODEL_PATH,
                MODEL_NAME,
                preprocessor.x_train,
                preprocessor.y_train,
            )
            model.compile_and_fit(epochs=1, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE)
            # Test model with first sample of test data
            x_test_sample = preprocessor.x_test
            prediction = model.predict(preprocessor.x_test, scaler=preprocessor.scaler[preprocessor.target])
            # Plot test and prediction, reset plot first
            plt.cla()
            plt.clf()
            plt.plot(preprocessor.y_test, label="Test", color="green")
            plt.plot(prediction, label="Prediction", color="blue")
            plt.legend()
            plt.grid()
            plt.savefig(f"{PATH_TEST_RESULTS}/{pair}.png")

if __name__ == "__main__":
    unittest.main()