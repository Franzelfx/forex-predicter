"""System test for all modules."""
import unittest
import numpy as np
import pandas as pd
from config_tb import *
from model import Model
from indicators import Indicators
from matplotlib import pyplot as plt
from data_aquirer import Data_Aquirer
from preprocessor import Preprocessor

class SystemTest(unittest.TestCase):
    """Test the system."""

    def _test_system(self):
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
                    MODEL_NAME,
                    preprocessor.x_train,
                    preprocessor.y_train,
                )
                model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE, learning_rate=TEST_LEARNING_RATE)
                x_test = preprocessor.x_test
                if TEST_SCALE is True:
                    prediction = model.predict((x_test))
                else:
                    prediction = model.predict(x_test).flatten()
                # Reduce to time_steps_out
                prediction = prediction[:TEST_TIME_STEPS_OUT]
                # Plot the results
                plt.cla()
                plt.clf()
                # Plot prediction and actual values
                y_test = preprocessor.y_test[:TEST_TIME_STEPS_OUT]
                # Scale y_test back to original scale
                if TEST_SCALE is True:
                    scaler = preprocessor.scaler[preprocessor.target]
                    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                prediction = self.moving_average(prediction, 5)
                # Set y_test to same length as prediction (cut first values)
                y_test = y_test[-len(prediction):]
                # Plot the results
                # Set high dpi
                plt.style.use('dark_background')
                plt.rcParams["figure.dpi"] = 1200
                plt.plot(prediction, label="prediction")
                plt.plot(y_test, label="actual")
                plt.legend()
                plt.title(f"Prediction for {MODEL_NAME}")
                plt.xlabel("Time")
                plt.ylabel("Value")
                # Save the plot
                plt.savefig(f"{MODEL_PATH}/model_test/{MODEL_NAME}_test.png")
                # Save raw data as csv
                df = pd.DataFrame({"prediction": prediction, "actual": y_test})
                df.to_csv(f"{MODEL_PATH}/model_test/{MODEL_NAME}_test.csv", index=False)
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