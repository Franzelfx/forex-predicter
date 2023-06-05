"""Testbench for the model class."""
import unittest
import pandas as pd
from config_tb import *
from src.visualizer import Visualizer
from src.indicators import Indicators
from src.preprocessor import Preprocessor
from src.data_aquirer import Data_Aquirer
from src.time_series_model import TimeSeriesModel as Model

class Test_Model(unittest.TestCase):
    """Integration test for the Model class.

    @remarks This test is not a unit test, but an integration test. It tests the
                preprocessor and model class together.
    """

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        try:
            test_data = pd.read_csv(MODEL_DATA_SOURCE)
        except:
            aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
            test_data = aquirer.get(
                PAIR, MINUTES, start=START, save=True, from_file=False
            )
            # Indicators
            indicators = Indicators(test_data, TEST_INDICATORS)
            test_data = indicators.calculate(
                save=True, path=f"{PATH_INDICATORS}/{PAIR}_{MINUTES}.csv"
            )
        preprocessor = Preprocessor(
            test_data,
            TARGET,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
            shift=TEST_SHIFT,
        )
        preprocessor.summary()
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Run for testing
        model.compile_and_fit(
            epochs=TEST_EPOCHS,
            hidden_neurons=TEST_NEURONS,
            batch_size=TEST_BATCH_SIZE,
            learning_rate=TEST_LEARNING_RATE,
            branched_model=TEST_BRANCHED_MODEL,
            validation_spilt=TEST_VALIDATION_SPLIT,
        )
        # Predict the next values
        x_test = preprocessor.x_test
        prediction = model.predict(x_test, scaler=preprocessor.target_scaler)
        # Reduce to time_steps_out
        prediction = prediction[:TEST_TIME_STEPS_OUT]
        y_test = preprocessor.y_test[:TEST_TIME_STEPS_OUT]
        if TEST_SCALE:
            # Inverse the scaling
            scaler = preprocessor.target_scaler
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        # Plot the results
        visualizer = Visualizer(PAIR)
        path = f"{MODEL_PATH}/model_test"
        visualizer.plot_prediction(prediction, path, y_test=y_test)
        

if __name__ == "__main__":
    unittest.main()
