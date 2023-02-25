"""Plot the data from the csv file"""
import sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config_tb import *
from matplotlib import pyplot as plt

PERIOD = 30

def moving_average(data: pd.DataFrame, n: int):
    """Calculate the moving average for the given data."""
    data = np.array(data)
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def diff(pred: pd.DataFrame, actual: pd.DataFrame):
    """Calculate the difference betwen the first actual value and the first predicted value."""
    return -(actual[0] - pred[0])

def csv_plot(pair: str):
    plt.cla()
    plt.clf()
    data = pd.read_csv(f"test/system_test/{pair}_prediction.csv")
    plt.style.use('dark_background')
    avg = moving_average(data['prediction'], PERIOD)
    # Substract diff
    avg = avg - diff(avg, data['actual'])
    # Plot moving average prediction (shiftet by 200 to the right)
    plt.plot(range(PERIOD,len(avg) + PERIOD),avg, label="prediction")
    # Plot actual values
    plt.plot(data['actual'], label="actual")
    plt.legend()
    plt.grid(color='gray', linewidth=0.5)
    plt.savefig(f"test/system_test/{pair}_test_plot.png", dpi=300)


for pair in REQUEST_PAIRS:
    csv_plot(pair)