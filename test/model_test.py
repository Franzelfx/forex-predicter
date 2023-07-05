"""Testbench for the model class."""
import logging
import unittest
import pandas as pd
import tensorflow as tf
from config_tb import *
from src.model import Model
from src.indicators import Indicators
from src.preprocessor import Preprocessor
from src.data_aquirer import Data_Aquirer


class Test_Model(unittest.TestCase):
    """Integration test for the Model class.

    @remarks This test is not a unit test, but an integration test. It tests the
                preprocessor and model class together.
    """

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        pair = os.environ.get("START_PAIR")
        from_saved_file = os.environ.get("FROM_SAVED_FILE")
        use_multiple_gpus = os.environ.get("USE_MULTIPLE_GPUS")
        # Data
        CORR_PAIRS = ["C:USDCHF", "C:EURCAD", "C:GBPJPY"]
        aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
        api_data = aquirer.get(
            pair, MINUTES_TRAIN, start=START, end=END, save=True, from_file=from_saved_file
        )
        indicators = Indicators(PATH_INDICATORS, pair, api_data, TEST_INDICATORS)
        indicator_data = indicators.calculate(save=True)
        # Get all correlated pairs
        for corr_pair in CORR_PAIRS:
            api_data_corr = aquirer.get(
                corr_pair,
                MINUTES_TRAIN,
                start=START,
                end=END,
                save=True,
                from_file=from_saved_file,
            )
            # Apply indicator to correlated pair
            indicators = Indicators(
                PATH_INDICATORS, corr_pair, api_data_corr, TEST_INDICATORS
            )
            indicator_data_corr = indicators.calculate(save=False)
            # Add pair:name to column_names
            indicator_data_corr.columns = [
                f"{corr_pair[2:]}:{column}" for column in indicator_data_corr.columns
            ]
            # Drop all rows where the dates do not match
            indicator_data_corr = indicator_data_corr[
                indicator_data_corr.index.isin(indicator_data.index)
            ]
            # Append correlated pair to indicator_data
            indicator_data_corr = pd.concat(
                [indicator_data, indicator_data_corr], axis=1, sort=False
                )
        # Concatenate both dataframes
        indicator_data = pd.concat([indicator_data, indicator_data_corr], axis=1, sort=False)
        # Preprocess data
        preprocessor = Preprocessor(
            indicator_data,
            TARGET,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            scale=TEST_SCALE,
            shift=TEST_SHIFT,
        )
        preprocessor.summary()
        # Model
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            preprocessor.x_train,
            preprocessor.y_train,
        )
        # Run for testing
        if use_multiple_gpus:
            strategy = tf.distribute.MirroredStrategy()
        model.compile(
            learning_rate=TEST_LEARNING_RATE,
            hidden_neurons=TEST_NEURONS,
            strategy=strategy if use_multiple_gpus == 'True' else None,
        )
        model.fit(
            epochs=TEST_EPOCHS,
            batch_size=TEST_BATCH_SIZE,
            validation_split=TEST_VALIDATION_SPLIT,
            patience=TEST_PATIENCE,
        )


if __name__ == "__main__":
    # get API_KEY from environment variable
    API_KEY = os.environ.get("API_KEY")
    try:
        unittest.main()
    except Exception as e:
        logging.error(e)
