"""The utilizer module, to use the trained model to predict."""
import numpy as np
from src.preprocessor import Preprocessor
from src.model import Model as ModelPreTrained
from sklearn.preprocessing import MinMaxScaler


class Utilizer():
    """The utilizer class, to use the trained model to predict."""

    def __init__(self, model: ModelPreTrained, preprocessor: Preprocessor, ma_period=20, lookback=3) -> None:
        """Initialize the utilizer.
        
        @param model The model to use for prediction or the path to the model.
        @param data The data to use for prediction or the path to the data.
        """
        self._model = model
        self._preprocessor = preprocessor
        self._ma_period = ma_period
        self._lookback = lookback
        # Check if model is a path
        if isinstance(model, str):
            self._model = ModelPreTrained.load(model)
    
    @property
    def test_actual(self) -> np.ndarray:
        """Get the actual test values.
        
        @return The actual test values.
        """
        # Return actual, wuth ma_period values removed at beginning
        return self._preprocessor.y_test_inverse[self._ma_period:]
    
    @property
    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Get test and hat prediction.
        
        @return Tuple of test and hat prediction.
        """
        # Predict the values
        test = self._model.predict(self._preprocessor.x_test, scaler=self._preprocessor.target_scaler, from_saved_model=True)
        y_hat = self._model.predict(self._preprocessor.x_hat, scaler=self._preprocessor.target_scaler, from_saved_model=True)
        # Calculate moving average
        test = self._moving_average(test, self._ma_period)
        y_hat = self._moving_average(y_hat, self._ma_period)
        # Substract the difference
        test = test + self._diff(test, self._preprocessor.first_known_y)
        y_hat = y_hat + self._diff(y_hat, self._preprocessor.last_known_y)
        return test, y_hat
    
    def _mean_of(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate the mean of the given data.
        
        @param data The data to calculate the mean for.
        @param period The period to calculate the mean for.
        @return Array of new values.
        """
        mean = []
        for i in range(0, len(data), period):
            mean.append(np.mean(data[i:i+period]))
        return np.array(mean)

    def _moving_average(self, data: np.ndarray, n: int) -> np.ndarray:
        """Calculate the moving average for the given data.
        
        @param data The data to calculate the moving average for.
        @param n The number of values to average.
        @return The moving average.
        """
        data = np.array(data)
        ret = np.cumsum(data, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def _diff(self, pred: np.ndarray, actual: int) -> np.ndarray:
        """Calculate the difference betwen the first actual value and the first predicted value.
        
        @param pred The predicted values.
        @param actual The actual values.
        @return The difference.
        """
        return -(actual - pred[0])
