"""System test for all modules."""
import unittest
import numpy as np
import pandas as pd
from config_tb import *
from src.model import Model
from src.indicators import Indicators
from matplotlib import pyplot as plt
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
                print(preprocessor.data.head(5))
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
                plt.cla()
                plt.clf()
                plt.plot(prediction, label="prediction")
                plt.plot(y_test, label="actual")
                plt.legend()
                plt.title(f"Prediction for {pair}")
                plt.xlabel("Time")
                plt.ylabel("Value")
                # Save the plot
                plt.savefig(f"{pair}/model_test/{pair}_test.png", dpi=600)
                # Save raw data as csv
                df = pd.DataFrame({"prediction": prediction, "actual": y_test})
                df.to_csv(f"{MODEL_PATH}/model_test/{pair}_test.csv", index=False)
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