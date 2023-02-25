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
    def plot_prediction(self, prediction, path, y_test=None, save_csv=True, extra_info=""):
        """Plot the prediction."""
        date = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"{path}/{self.pair}_prediction_{extra_info}"
        plt.cla()
        plt.clf()
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        plt.plot(prediction, label="prediction")
        if y_test is not None:
            plt.plot(y_test, label="actual")
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
            df = pd.DataFrame({"prediction": prediction, "actual": y_test})
            df.to_csv(f"{path}.csv", index=False, )
