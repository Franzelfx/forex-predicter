"""Testbench for the model class."""
import unittest
import pandas as pd
from config_tb import *
from model import Model
from matplotlib import pyplot as plt
from preprocessor import Preprocessor

class Test_Model(unittest.TestCase):
    """Integration test for the Model class.

    @remarks This test is not a unit test, but an integration test. It tests the
                preprocessor and model class together.
    """

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        test_data = pd.read_csv(MODEL_DATA_SOURCE)
        # Remove 'o', 'h', 'l', 'c', 'v', 'vw' columns
        test_data = test_data.drop(columns=['o', 'h', 'l', 'c', 'v', 'vw'])
        preprocessor = Preprocessor(
            test_data,
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
        # Run 30 epochs for testing
        model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE, learning_rate=TEST_LEARNING_RATE)
        # Predict the next values

        # predict last sample of x_train
        #prediction_train = model.predict(preprocessor.x_train)
        # x_test is only the first sample of x_test
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
        # Plot the results
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

if __name__ == "__main__":
    unittest.main()
