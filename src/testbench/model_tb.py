"""Testbench for the model class."""
import unittest
import numpy as np
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

    # def test_compile_and_fit(self):
    #     """Test the compile and fit method with data from the preprocessor."""
    #     test_data = pd.read_csv(MODEL_DATA_SOURCE)
    #     preprocessor = Preprocessor(
    #         test_data,
    #         "c",
    #         test_split=TEST_SPLIT,
    #         time_steps_in=TEST_TIME_STEPS_IN,
    #         time_steps_out=TEST_TIME_STEPS_OUT,
    #         intersection_factor=TEST_INTERSECTION_FACTOR,
    #         scale=TEST_SCALE,
    #     )
    #     model = Model(
    #         MODEL_PATH,
    #         MODEL_NAME,
    #         preprocessor.x_train,
    #         preprocessor.y_train,
    #     )
    #     # Run only 10 epochs for testing
    #     fit = model.compile_and_fit(epochs=TEST_EPOCHS)
    #     print(fit.history.keys())

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        test_data = pd.read_csv(MODEL_DATA_SOURCE)
        preprocessor = Preprocessor(
            test_data,
            "c",
            test_split=TEST_SPLIT,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            intersection_factor=TEST_INTERSECTION_FACTOR,
            scale=TEST_SCALE,
        )
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Run 30 epochs for testing
        fit = model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE)
        # Predict the next values
        print(preprocessor.x_test.shape)
        prediction = model.predict(preprocessor.x_test, scaler=preprocessor.scaler[preprocessor.target])
        # Plot test and prediction, reset plot first
        plt.cla()
        plt.clf()
        plt.plot(preprocessor.y_test, label="Test", color="green")
        plt.plot(prediction, label="Prediction", color="blue")
        plt.legend()
        plt.grid()
        plt.show()
        


if __name__ == "__main__":
    unittest.main()
