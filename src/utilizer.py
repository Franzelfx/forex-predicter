"""The utilizer module, to use the trained model to predict."""
import numpy as np
from logging import warning
from src.preprocessor import Preprocessor
from src.model import Model as ModelPreTrained


class Utilizer:
    """The utilizer class, to use the trained model to predict."""

    def __init__(self, model: ModelPreTrained, preprocessor: Preprocessor) -> None:
        """Initialize the utilizer.

        @param model The model to use for prediction or the path to the model.
        @param data The data to use for prediction or the path to the data.
        """
        self._model = model
        self._preprocessor = preprocessor
        # Check if model is a path
        if isinstance(model, str):
            self._model = ModelPreTrained.load(model)

    @property
    def test_actual(self) -> np.ndarray:
        """Get the actual test values.

        @return The actual test values.
        """
        return self._preprocessor.y_test_inverse

    def predict(self, box_pts=0, lookback=None) -> tuple[np.ndarray, np.ndarray]:
        """Get test and hat prediction.

        @return Tuple of test and hat prediction.
        """
        # Predict the values
        # Predict the values for each sequence
        print(f"Predicting with a lookback of {lookback}.")
        y_train, y_test, y_hat = self._model.predict(
            self._preprocessor.x_hat,
            x_train=x_train,
            x_test=x_test,
            scaler=self._preprocessor.target_scaler,
            from_saved_model=True,
        )
        print(y_hat)
        # Substract the difference
        # Warning, if x_test and x_hat are the same
        if np.array_equal(self._preprocessor.x_test, self._preprocessor.x_hat):
            warning("x_test and x_hat are the same")
        first_actual = self.test_actual[0]
        #y_test = y_test - self._diff(y_test, first_actual)
        y_hat = y_hat - self._diff(y_hat, self._preprocessor.last_known_y)
        # Smooth the data
        if box_pts > 0:
            y_test = self._concat_moving_average(
                self._preprocessor.x_test_target_inverse, y_test, box_pts
            )
            y_hat = self._concat_moving_average(
                self._preprocessor.x_hat_target_inverse, y_hat, box_pts
            )
        return y_test, y_hat

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
