"""System test for all modules."""
import unittest
from config_tb import *
from model import Model
from indicators import Indicators
from data_aquirer import Data_Aquirer
from preprocessor import Preprocessor
from matplotlib import pyplot as plt
import pandas as pd

class SystemTest(unittest.TestCase):
    """Test the system."""

    def test_system(self):
        """Test the system."""
        # Get data for all pairs
        multiple_pairs: pd.DataFrame = pd.DataFrame()
        for pair in REQUEST_PAIRS:
            aquirer = Data_Aquirer(PATH_PAIRS, API_KEY, api_type='full')
            data = aquirer.get(pair, MINUTES, save=True)
            # Apply indicators
            indicators = Indicators(data, TEST_INDICATORS)
            data = indicators.calculate(save=True, path=f"{PATH_INDICATORS}/{pair}_{MINUTES}.csv")
            # Add pair name to column name e.g. 'c_BTCUSD'
            data.columns = [f"{col}_{pair}" for col in data.columns]
            # Concat dataframes
            multiple_pairs = pd.concat([multiple_pairs, data], axis=1)
            # drop index
            multiple_pairs.reset_index(drop=True, inplace=True)
        
        # Preprocess data
        preprocessor = Preprocessor(multiple_pairs, f"{TARGET}_{PAIR}")
        print(preprocessor.data.head(5))
        preprocessor.summary()
        # Define model
        model = Model(MODEL_PATH, MODEL_NAME, preprocessor.x_train, preprocessor.y_train)
        # Train model
        model.compile_and_fit(epochs=TEST_EPOCHS, hidden_neurons=TEST_NEURONS, batch_size=TEST_BATCH_SIZE, learning_rate=TEST_LEARNING_RATE)
        # Predict the next values
        x_test = preprocessor.x_test
        if TEST_SCALE is True:
            prediction = model.predict((x_test))
        else:
            prediction = model.predict(x_test).flatten()
        # Reduce to time_steps_out
        prediction = prediction[:TEST_TIME_STEPS_OUT]
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
        plt.savefig(f"{MODEL_PATH}/system_test/{MODEL_NAME}_test.png")
        # Save raw data as csv
        df = pd.DataFrame({"prediction": prediction, "actual": y_test})
        df.to_csv(f"{MODEL_PATH}/system_test/{MODEL_NAME}_test.csv", index=False)

if __name__ == "__main__":
    unittest.main()