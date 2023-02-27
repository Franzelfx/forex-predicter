"""The utilizer module, to use the trained model to predict."""
import numpy as np
from src.model import Model as ModelPreTrained
from sklearn.preprocessing import MinMaxScaler


class Utilizer():
    """The utilizer class, to use the trained model to predict."""

    #TODO: EDIT README and add Model | str to the model parameter (python 3.10)
    def __init__(self, model, data) -> None:
        """Initialize the utilizer.
        
        @param model The model to use for prediction or the path to the model.
        @param data The data to use for prediction or the path to the data.
        """
        self._model: ModelPreTrained = model
        self._data = data
        # Check if model is a path
        if isinstance(model, str):
            self._model = ModelPreTrained.load(model)

    def moving_average(self, data: np.ndarray, n: int) -> np.ndarray:
        """Calculate the moving average for the given data.
        
        @param data The data to calculate the moving average for.
        @param n The number of values to average.
        @return The moving average.
        """
        data = np.array(data)
        ret = np.cumsum(data, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    def diff(self, pred: np.ndarray, actual: int) -> np.ndarray:
        """Calculate the difference betwen the first actual value and the first predicted value.
        
        @param pred The predicted values.
        @param actual The actual values.
        @return The difference.
        """
        return -(actual - pred[0])

    def predict(self, time_steps: int, scaler: MinMaxScaler=None, last_known: int=None, ma_period=50) -> np.ndarray:
        """Predict the next values.
        
        @param time_steps The number of time steps to predict.
        @param scaler The scaler to use for the prediction.
        @param smoothen The number of values to smoothen (moving average).
        @param last_known The last known value(s) to use for prediction.
        @return The predicted values.
        """
        # Get last time_steps values of data
        data = self._data[-time_steps:]
        # Predict the next values
        prediction = self._model.predict(data, scaler=scaler, from_saved_model=True)
        # Smoothen the prediction
        prediction = self.moving_average(prediction, ma_period)
        # Substract diff
        if last_known is not None:
            prediction = prediction - self.diff(prediction, last_known)
        return prediction
