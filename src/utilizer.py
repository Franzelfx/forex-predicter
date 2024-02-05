"""The utilizer module, to use the trained model to predict."""
import os
import csv
import numpy as np
import pandas as pd
from typing import List
from src.preprocessor import Preprocessor
from src.model import Model as ModelPreTrained
# Logging
from src.logger import logger as loguru

PATH = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.abspath(os.path.join(PATH, "metrics"))

class Utilizer:
    """The utilizer class, to use the trained model to predict."""

    def __init__(self, model: ModelPreTrained, preprocessor: Preprocessor | List) -> None:
        """Initialize the utilizer.

        @param model The model to use for prediction or the path to the model.
        @param preprocessor The preprocessor to use for prediction.
        """
        self._model = model
        self._y_hat: np.ndarray = np.array([])
        self._confidence: float = 0.0
        self._preprocessor = preprocessor
        # Check if model is a path
        if isinstance(model, str):
            self._model = ModelPreTrained.load(model)
        if isinstance(preprocessor, list):
            self._target_preprocessor = preprocessor[0]
            self._target_scaler = self._target_preprocessor.target_scaler
        else:
            self._target_scaler = preprocessor.target_scaler

    @property
    def test_actual(self) -> np.ndarray:
        """Get the actual test values.

        @return The actual test values.
        """
        return self._target_preprocessor.y_test_inverse
    
    @property
    def x_target(self) -> np.ndarray:
        """Get the target x values.

        @return The target x values.
        """
        return self._target_preprocessor.x_hat_target_inverse
    
    @property
    def confidence(self) -> str:
        """Get the confidence score.

        @return The confidence score.
        """
        return f"{self._confidence} %"
    
    def evaluate(self, x_target, y_test) -> float:
        """Calculate the MAPE of x_target (actual values) and y_test (predicted values).

        Returns:
            float: MAPE of the model.
        """
        # Adjusting x_target to match the length of y_test
        x_target_adjusted = x_target[-len(y_test):]
        
        ape = np.abs((x_target_adjusted - y_test) / x_target_adjusted)
        return np.mean(ape) * 100

    def predict(self, box_pts=0, test=False) -> tuple[np.ndarray, np.ndarray]:
        """Get test and hat prediction.

        @return Tuple of test and hat prediction.
        """
        # Predict the values
        # Predict the values for each sequence
        x_hat = []
        x_test = []
        if isinstance(self._preprocessor, list):
            for preprocessor in self._preprocessor:
                x_hat.append(preprocessor.x_hat)
                x_test.append(preprocessor.x_test)
        else:
            x_hat = self._preprocessor.x_hat
            x_test = self._preprocessor.x_test
        # Predict the values 
        y_test, y_hat = self._model.predict(x_hat, self._target_scaler, from_saved_model=True, x_test=x_test if test else None)
        # Calculate difference for y_hat and all y_test values
        y_hat = y_hat - self._diff(y_hat, self._target_preprocessor.last_known_y)
        if test:
            #y_test = y_test - self._diff(y_test, self._target_preprocessor.y_test[0])
            self._store_correlaction_confidence(self._target_preprocessor.y_test_inverse, y_test)
        # Smooth the data
        if box_pts > 0:
             y_hat = self._concat_moving_average(
                 self._target_preprocessor.x_hat_target_inverse, y_hat, box_pts
             )
             if test:
                    y_test = self._concat_moving_average(
                        self._target_preprocessor.x_test_target_inverse, y_test, box_pts
                    )
        # Calculate confidence score
        self._confidence = self._model.confidence(x_hat)
        loguru.info(f"Confidence score: {self.confidence}")
        self._y_hat = y_hat
        return y_test, y_hat

    def _store_correlaction_confidence(self, y_actual, y_hat):
        # Convert to pandas Series
        y_actual_series = pd.Series(y_actual.flatten())
        y_hat_series = pd.Series(y_hat.flatten())

        # Check if either series is empty or has insufficient data
        if y_actual_series.empty or y_hat_series.empty or len(y_actual_series) < 2 or len(y_hat_series) < 2:
            loguru.warning("Insufficient data for correlation calculation.")
            return None

        # Calculate correlation
        corr = y_actual_series.corr(y_hat_series)
        loguru.info(f"Calculated correlation: {corr}")

        # Calculate confidence correlation
        corr_conf = corr * self._confidence
        loguru.info(f"Calculated confidence correlation: {corr_conf}")

        # Save the correlation confidence, confidence, and correlation in a CSV file
        csv_file_path = os.path.join(CSV_PATH, "corr_conf.csv")
        try:
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Correlation', 'Confidence', 'Correlation Confidence'])
                writer.writerow([corr, self._confidence, corr_conf])
            loguru.info(f"Correlation confidence saved successfully at {csv_file_path}")
        except Exception as e:
            loguru.error(f"Error in saving correlation confidence: {e}")

        return corr_conf


    def _concat_moving_average(
        self, x_hat: np.ndarray, y_hat: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate the mean of the given data.

        @param data The data to calculate the mean for.
        @param period The period to calculate the mean for.
        @return Array of new values.
        """
        concat = np.concatenate((x_hat, y_hat), axis=0)
        concat = self._moving_average(concat, period)
        return concat[-len(y_hat) :]

    def _moving_average(self, data: np.ndarray, n: int) -> np.ndarray:
        """Calculate the moving average for the given data.

        @param data The data to calculate the moving average for.
        @param n The number of values to average.
        @return The moving average.
        """
        data = np.array(data)
        ret = np.cumsum(data, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    def _smooth(self, data: np.ndarray, n: int) -> np.ndarray:
        """Smooth the given data.

        @param data The data to smooth.
        @param n The number of values to average.
        @return The smoothed data.
        """
        box = np.ones(n) / n
        return np.convolve(data, box, mode="same")

    def _diff(self, pred: np.ndarray, actual: int) -> np.ndarray:
        """Calculate the difference betwen the first actual value and the first predicted value.

        @param pred The predicted values.
        @param actual The actual values.
        @return The difference.
        """
        return -(actual - pred[0])
