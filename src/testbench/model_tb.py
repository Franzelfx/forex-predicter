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
        # Only keep 'MA50' 
        test_data = test_data[['MA50']]
        preprocessor = Preprocessor(
            test_data,
            TARGET,
            test_split=TEST_SPLIT,
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
        model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE)
        # Predict the next values

        # predict last sample of x_train
        #prediction_train = model.predict(preprocessor.x_train)
        if TEST_SCALE is True:
            prediction = model.predict(preprocessor.x_test, scaler=preprocessor.scaler[preprocessor.target])
            prediction_train = model.predict(preprocessor.x_train, scaler=preprocessor.scaler[preprocessor.target])
        else:
            prediction = model.predict(preprocessor.x_test)
            prediction_train = model.predict(preprocessor.x_train)
        plt.cla()
        plt.clf()
        # Get last time_steps_in values from train_data
        train_data = preprocessor.train_data[preprocessor.target].values[-preprocessor.time_steps_in:]   
        # Plot train data
        plt.plot(train_data, label="Train", color="red")
        plt.plot(prediction_train[-len(train_data):], label="Prediction_Train", color="blue")
        # Plot y_test and prediction and shift them to the right
        # Get last ""steps_in" values from x_test
        x_test = preprocessor.x_test[:, :, preprocessor.loc_of(TARGET)]
        x_test = x_test.flatten()
        # Arrange some x to shift the plot to the right
        x = range(len(train_data), len(train_data) + len(x_test))
        plt.plot(x, x_test, label="x_test", color="green")
        # Arrange some x to shift the plot to the right
        x_start = len(train_data) + preprocessor.time_steps_in
        x_end = len(train_data) + preprocessor.time_steps_in + len(preprocessor.y_test)
        x = range(x_start, x_end)
        # Plot prediction and y_test (shifted to the right)
        plt.plot(x, prediction, label="Prediction", color="blue")
        plt.plot(x, preprocessor.y_test, label="y_test", color="orange")
        plt.legend()
        plt.grid()
        plt.savefig(f"{MODEL_PATH}/model_test/{MODEL_NAME}_test.png")
        plt.show()
        


if __name__ == "__main__":
    unittest.main()
