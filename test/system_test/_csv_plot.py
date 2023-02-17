"""Plot the data from the csv file"""
import sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config_tb import *
from matplotlib import pyplot as plt



def moving_average(data, n):
    """Calculate the moving average for the given data."""
    data = np.array(data)
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def csv_plot(pair:str):
    plt.cla()
    plt.clf()
    data = pd.read_csv(f"test/system_test/{pair}_{MINUTES}_{pair}_prediction.csv")
    plt.style.use('dark_background')
    avg = moving_average(data['prediction'], 200)
    # Plot moving average prediction (shiftet by 200 to the right)
    plt.plot(range(200,len(avg) + 200),avg, label="prediction")
    # Plot actual values
    plt.plot(data['actual'], label="actual")
    plt.legend()
    plt.grid(color='gray', linewidth=0.5)
    plt.savefig(f"test/sytem_test/{pair}_test_plot.png", dpi=300)


for pair in REQUEST_PAIRS:
    csv_plot(pair)