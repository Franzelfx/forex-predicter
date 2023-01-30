"""Config file for testbench."""
import sys
import os.path
from datetime import datetime as date


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Data aquirer
PATH_PAIRS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, "pairs"))
PAIR = "EURUSD"
MINUTES = 15
API_KEY = "kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo"
TIME_FORMAT = "%Y-%m-%d"
DATE_START = "2022-10-01"
DATE_END = date.today().strftime(TIME_FORMAT)
API_TYPE = "basic"

# Indicators
PATH_INDICATORS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, f"indicators/{PAIR}_{MINUTES}.csv"))
INDICATORS_DATA_SOURCE = f"{PATH_PAIRS}/{PAIR}_{MINUTES}.csv"
TEST_INDICATORS = ["ATR", "BOLLINGER", "MA50", "MA200", "MACD", "RSI", "STOCHASTIC"]
EXPECTED_COLUMNS = [
    "v",
    "vw",
    "o",
    "c",
    "h",
    "l",
    "n",
    "ATR",
    "BOLLINGER_UPPER",
    "BOLLINGER_MIDDLE",
    "BOLLINGER_LOWER",
    "MA50",
    "MA200",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "RSI",
    "STOCHASTIC_K",
    "STOCHASTIC_D",
]

# Preprocessor
PREPROCESSOR_DATA_SOURCE = f"{PATH_INDICATORS}"
TEST_SPLIT = 0.2
TEST_TIME_STEPS_IN = 60
TEST_TIME_STEPS_OUT = 30
TEST_INTERSECTION_FACTOR = 0.1
TEST_SCALE = True

# Model
MODEL_DATA_SOURCE = f"{PATH_INDICATORS}"
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_NAME = 'EURUSD_15'
TEST_EPOCHS = 10