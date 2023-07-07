"""Testbench for the model class."""
import logging
import unittest
import numpy as np
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

    def synchronize_dataframes(dataframes, column_to_sync='t'):
        # Create a list to hold all synchronized dataframes
        synced_dataframes = []
        
        # Start with the first dataframe
        synced_df = dataframes[0]

        # Loop through the rest of the dataframes
        for df in dataframes[1:]:
            # Merge on the column_to_sync with an inner join
            synced_df = pd.merge(synced_df, df, how='inner', on=column_to_sync)
            
        # Now we have a dataframe that contains rows with 't' values 
        # that appear in all original dataframes. Next, we need to
        # create a synchronized version of each original dataframe.
        
        # Loop through the original dataframes again
        for df in dataframes:
            # For each dataframe, keep only the rows that exist in synced_df
            synced_dataframes.append(df[df[column_to_sync].isin(synced_df[column_to_sync])])
            
        return synced_dataframes

    def test_compile_fit_predict(self):
        """Test the compile, fit and predict method with data from the preprocessor."""
        pair = os.environ.get("START_PAIR")
        from_saved_file = os.environ.get("FROM_SAVED_FILE")
        use_multiple_gpus = os.environ.get("USE_MULTIPLE_GPUS")
        # Data
        CORR_PAIRS = ["C:USDCHF", "C:EURCAD", "C:GBPJPY"]
        pairs = []
        # First get the target pair
        aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type="full")
        target_pair = aquirer.get(
            pair, MINUTES_TRAIN, start=START, end=END, save=True, from_file=from_saved_file
        )
        indicators = Indicators(PATH_INDICATORS, pair, target_pair, TEST_INDICATORS)
        indicator_data = indicators.calculate(save=True)
        pairs.append(indicator_data)
        # Get correlated pairs
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
            # get a series of unique values from the 't' column of indicator_data
            values_to_keep = indicator_data['t'].unique()
            # filter rows of indicator_data_corr, keeping only those where 't' exists in values_to_keep
            indicator_data_corr = indicator_data_corr[indicator_data_corr['t'].isin(values_to_keep)]
            # print the length of indicator_data_corr
            print(f"Length of indicator_data_corr: {len(indicator_data_corr)}")
            # rename all columns except 't'
            indicator_data_corr.columns = [
                f"{col}_{corr_pair}" if col != 't' else col for col in indicator_data_corr.columns
            ]
            # Preprocess the correlated pair
            pairs.append(indicator_data_corr)
            # Append to the list of correlated pairs
        # Synchronize the dataframes
        synced_df = Test_Model.synchronize_dataframes(pairs)
        # Target pair is first dataframe of the list
        target_pair = synced_df[0]
        # Correlated pairs are the rest of the dataframes
        correlated_pairs = synced_df[1:]
        # Preprocessor for target pair
        target_pair = Preprocessor(
            PREPROCESSOR_PATH,
            time_steps_in=TEST_TIME_STEPS_IN,
            time_steps_out=TEST_TIME_STEPS_OUT,
            test_length=TEST_LENGTH,
            target=TARGET,
            scale=TEST_SCALE,
            shift=TEST_SHIFT,
        )
        # Preprocessor for correlated pairs
        corr_pairs = []
        for corr_pair in correlated_pairs:
            corr_pair = Preprocessor(
                PREPROCESSOR_PATH,
                time_steps_in=TEST_TIME_STEPS_IN,
                time_steps_out=TEST_TIME_STEPS_OUT,
                test_length=TEST_LENGTH,
                scale=TEST_SCALE,
                shift=TEST_SHIFT,
            )
            corr_pairs.append(corr_pair)

        # Model
        model = Model(
            MODEL_PATH,
            MODEL_NAME,
            target_pair.y_train,
        )
        for corr_pair in corr_pairs:
            if isinstance(corr_pair.x_train, np.ndarray):
                model.add_branch(corr_pair.x_train, [64], [64], [64], [2], [0.2])
        model.summation([64], [0.2])
        model.output([64], [0.2])
        # Run for testing
        if use_multiple_gpus:
            strategy = tf.distribute.MirroredStrategy()
        model.compile(
            learning_rate=TEST_LEARNING_RATE,
            strategy=strategy if use_multiple_gpus == 'True' else None,
        )
        model.fit(
            epochs=TEST_EPOCHS,
            batch_size=TEST_BATCH_SIZE,
            validation_split=TEST_VALIDATION_SPLIT,
            patience=TEST_PATIENCE
        )


if __name__ == "__main__":
    # get API_KEY from environment variable
    API_KEY = os.environ.get("API_KEY")
    try:
        unittest.main()
    except Exception as e:
        logging.error(e)
