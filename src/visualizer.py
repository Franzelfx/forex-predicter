"""Visualize the results of the model."""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

class Visualizer:
    """Visualize the results of the model."""

    #TODO: Whole data as input, to get information about
    #      date, time, time iterations, etc.
    def __init__(self, target: str, dark_mode: bool = True):
        """Initialize the visualizer."""
        self.pair = target
        self.dark_mode = dark_mode

    # TODO: Add time information to x-axis
    def plot_prediction(self, hat, path, test_actual=None, test_predict=None, save_csv=True, extra_info=""):
        """Plot the prediction."""
        date = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        if extra_info != "":
            extra_info = f"_{extra_info}"
        path = f"{path}/{self.pair}_prediction{extra_info}"
        plt.cla()
        plt.clf()
        shift_len = 0
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        if (test_actual.all() or test_predict.all()) is not None:
            if(test_actual.all() is not None):
                plt.plot(test_actual, label="Actual")
                shift_len = len(test_actual)
            if(test_predict.all() is not None):
                plt.plot(test_predict, label="Prediction")
                shift_len = len(test_predict)
            # Then we have to shift the prediction
            # by the length of the input to the right
            # to get the correct time
            plt.plot(range(shift_len, shift_len + len(hat)), hat, label="Prediction")
        else:
            plt.plot(hat, label="Prediction")
        plt.legend()
        plt.title(f"Prediction for {self.pair}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        # Set title (pair name and date)
        plt.title(f"{self.pair} {date}")
        # Save the plot
        plt.savefig(f"{path}.png", dpi=600)
        print(f"Saved plot to {path}.png")
        # Save raw data as csv
        if save_csv:
            df = pd.DataFrame({"prediction": hat})
            df.to_csv(f"{path}.csv", index=False, )
