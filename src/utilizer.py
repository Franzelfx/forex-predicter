"""The utilizer module, to use the trained model to predict."""
import numpy as np
from src.preprocessor import Preprocessor
from src.model import Model as ModelPreTrained
from sklearn.preprocessing import MinMaxScaler


class Utilizer():
    """The utilizer class, to use the trained model to predict."""

    def __init__(self, model: ModelPreTrained, preprocessor: Preprocessor, ma_period=10) -> None:
        """Initialize the utilizer.
        
        @param model The model to use for prediction or the path to the model.
        @param data The data to use for prediction or the path to the data.
        """
        self._model = model
        self._preprocessor = preprocessor
        self._ma_period = ma_period
        # Check if model is a path
        if isinstance(model, str):
            self._model = ModelPreTrained.load(model)

    def predict(self) -> np.ndarray:
        prediction = self._model.predict(
            self._preprocessor.prediction_set,
            scaler=self._preprocessor.target_scaler,
            from_saved_model=True,
        )
        # Calculate moving average
        prediction = self._moving_average(prediction, self._ma_period)
        # Substract difference between last known value and first predicted value
        prediction = prediction - self._diff(prediction, self._preprocessor.last_known_y)
        # Calculate mean of for time_steps_out values
        prediction = self._mean_of(prediction, self._preprocessor.time_steps_out)
        return prediction
    
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
