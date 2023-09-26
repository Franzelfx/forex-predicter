import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

class Visualizer:
    """Visualize the results of the model."""

    def __init__(self, target: str, dark_mode: bool = True):
        """Initialize the visualizer."""
        if ":" in target:
            target = target.split(":")[1]
        self.pair = target
        self.dark_mode = dark_mode

    def plot_prediction(self, path, hat: np.ndarray, test_actual: np.ndarray = None, test_predict: np.ndarray = None, save_csv=True, extra_info="", time_base=None):
        """Plot the prediction."""
        date = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        if extra_info != "":
            extra_info = f"_{extra_info}"
        path = f"{path}/{self.pair}_prediction{extra_info}"

        # Clear the plot
        plt.cla()
        plt.clf()

        # Set style
        if self.dark_mode:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        # Set line width
        plt.rcParams['lines.linewidth'] = 1

        # Plot the test data
        if test_actual is not None:
            plt.plot(test_actual, label="Actual")

        # Plot the test prediction
        if test_predict is not None:
            plt.plot(range(len(test_actual) - len(test_predict), len(test_actual)), test_predict, label="Test")

        # Plot the hat prediction
        if hat is not None:
            plt.plot(range(len(test_actual), len(test_actual) + len(hat)), hat, label="Prediction")

            # Add a trendline for hat
            z = np.polyfit(range(len(test_actual), len(test_actual) + len(hat)), hat, 1)
            p = np.poly1d(z)
            plt.plot(range(len(test_actual), len(test_actual) + len(hat)), p(range(len(test_actual), len(test_actual) + len(hat))), 'r--', label="Trendline")

        plt.legend()
        plt.title(f"Prediction for {self.pair}")
        plt.xlabel(f"Time ({ time_base if time_base is not None else '' })")
        plt.ylabel("Value")
        # Set title (pair name and date)
        plt.title(f"{self.pair} {date}")
        plt.grid()

        # Save the plot
        plt.savefig(f"{path}.png", dpi=600)
        print(f"Saved plot to {path}.png")

        # Save raw data as csv
        if save_csv:
            df = pd.DataFrame({"prediction": hat})
            if test_actual is not None:
                df["actual"] = test_actual[-len(hat):]  # align the lengths for simplicity in DataFrame
            if test_predict is not None:
                df["test"] = test_predict[-len(hat):]  # align the lengths
            df.to_csv(f"{path}.csv", index=False)
            print(f"Saved data to {path}.csv")
