"""Plot the data from the csv file"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def moving_average(data, n):
    """Calculate the moving average for the given data."""
    data = np.array(data)
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = pd.read_csv("src/testbench/model_test/EURUSD_15_test.csv")
avg = moving_average(data['prediction'], 10)
# Plot moving average prediction
plt.plot(avg, label="prediction")
# Plot actual values
plt.plot(data['actual'], label="actual")
plt.show()
