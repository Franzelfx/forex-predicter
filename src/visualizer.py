import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
from datetime import datetime as dt
# Logging
from src.logger import logger as loguru

PNG_PATH = os.path.join("src", "model_predictions", "visualizer")

class Visualizer:
    """Visualize the results of the model."""

    def __init__(self, target: str, dark_mode: bool = True):
        """Initialize the visualizer."""
        if ":" in target:
            target = target.split(":")[1]
        self.pair = target
        self.dark_mode = dark_mode

    def plot_prediction(
        self,
        path,
        n,
        m,
        x_test,
        x_hat,
        y_test,
        y_test_actual = None,
        y_hat=None,
        extra_info="",
        time_base=None,
        end_time=None,
    ):
        if extra_info != "":
            extra_info = f"_{extra_info}"
        # Clear the plot
        plt.cla()
        plt.clf()

        # Set style
        if self.dark_mode:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        # Set line width
        plt.rcParams["lines.linewidth"] = 0.5

        x_splits = (
            [x_test[i : i + n] for i in range(0, len(x_test), n)]
            if x_test is not None
            else []
        )
        y_splits = (
            [y_test[i : i + m] for i in range(0, len(y_test), m)]
            if y_test is not None
            else []
        )
        y_test_actual = (
            [y_test_actual[i : i + m] for i in range(0, len(y_test_actual), m)]
            if y_test_actual is not None
            else []
        )

        position = 0  # starting position for plotting

        for x, y, y_act in zip_longest(x_splits, y_splits, y_test_actual):
            if x is not None:
                # Plot the x sequence
                plt.plot(
                    range(position, position + len(x)),
                    x,
                    color="cornflowerblue",
                    label="x_test" if position == 0 else "",
                )
                position += len(x)
            if y is not None:
                # Plot the y sequence
                plt.plot(
                    range(position, position + len(y)),
                    y,
                    color="lightcoral",
                    label="y_test" if position == len(x) else "",
                )
                position += len(y)

        # Add x_hat to the plot if it exists
        if x_hat is not None:
            plt.plot(
                range(position, position + len(x_hat)),
                x_hat,
                color="mediumseagreen",
                label="x_hat",
            )
            position += len(x_hat)

        # Add y_hat to the plot if it exists
        if y_hat is not None:
            plt.plot(
                range(position, position + len(y_hat)),
                y_hat,
                color="orchid",
                label="y_hat",
            )

            # Plotting the trendline for y_hat
            x_vals = np.array(range(position - len(y_hat), position))
            m, b = np.polyfit(x_vals, y_hat, 1)  # m is slope, b is y-intercept
            plt.plot(x_vals, m * x_vals + b, "r--", label="y_hat trend")

        # Get the end time, use the current time if not provided
        if end_time is None:
            end_time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            # Ensure end_time has hours, minutes, and seconds
            if len(end_time.split("-")) < 6:
                # Append the current hours, minutes, and seconds if they're not provided
                end_time += dt.now().strftime("-%H-%M-%S")

        # Add the legend, title, and labels
        plt.legend()
        plt.title(f"Prediction for {self.pair} on {end_time}")
        plt.xlabel(f"Timebase { time_base if time_base is not None else '' } minutes")
        plt.ylabel("Value")
        plt.grid()

        # Save the plot
        plt.savefig(f"{PNG_PATH}.png", dpi=600)
        loguru.info(f"Saved plot to {PNG_PATH}.png")