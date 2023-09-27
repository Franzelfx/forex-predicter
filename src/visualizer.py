import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
from datetime import datetime as dt

class Visualizer:
    """Visualize the results of the model."""

    def __init__(self, target: str, dark_mode: bool = True):
        """Initialize the visualizer."""
        if ":" in target:
            target = target.split(":")[1]
        self.pair = target
        self.dark_mode = dark_mode

    def plot_prediction(self, path, x_test, x_hat, y_test, y_test_actual, y_hat, n, m, save_csv=True, extra_info="", time_base=None):
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
        plt.rcParams['lines.linewidth'] = 0.75

        x_splits = [x_test[i:i+n] for i in range(0, len(x_test), n)] if x_test is not None else []

        y_splits = [y_test[i:i+m] for i in range(0, len(y_test), m)] if y_test is not None else []
        y_splits_actual = [y_test_actual[i:i+m] for i in range(0, len(y_test_actual), m)] if y_test_actual is not None else []

        position = 0  # starting position for plotting

        for x, y, y_act in zip_longest(x_splits, y_splits, y_splits_actual):
            if x is not None:
                # Plot the x sequence
                plt.plot(range(position, position + len(x)), x, color='cornflowerblue', label='x_test' if position == 0 else "")
                position += len(x)
            if y is not None:
                # Plot the y sequence
                plt.plot(range(position, position + len(y)), y, color='lightcoral', label='y_test' if position == len(x) else "")
                position += len(y)
            if y_act is not None:
                # Optionally, plot y_test_actual if provided
                plt.plot(range(position, position + len(y_act)), y_act, linestyle='dashed', color='orange', label='y_test_actual' if position == len(x) else "")
                position += len(y_act)

        # Add x_hat to the end if it exists
        if x_hat is not None:
            plt.plot(range(position, position + len(x_hat)), x_hat, color='mediumseagreen', label="x_hat")
            position += len(x_hat)

        # Add y_hat to the end if it exists
        if y_hat is not None:
            plt.plot(range(position, position + len(y_hat)), y_hat, color='orchid', label="y_hat")
            x_vals = np.array(range(position, position + len(y_hat)))
            m, b = np.polyfit(x_vals, y_hat, 1)  # m is slope, b is y-intercept
            plt.plot(x_vals, m * x_vals + b, 'r--')  # <- Red dashed trendline

        plt.legend()
        plt.title(f"Prediction for {self.pair} on {date}")
        plt.xlabel(f"Time ({ time_base if time_base is not None else '' })")
        plt.ylabel("Value")
        plt.grid()

        # Save the plot
        plt.savefig(f"{path}.png", dpi=600)
        print(f"Saved plot to {path}.png")