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
DATE_START = "2009-01-01"
DATE_END = date.today().strftime(TIME_FORMAT)
API_TYPE = "basic"

# Indicators
PATH_INDICATORS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, f"indicators"))
INDICATORS_DATA_SOURCE = f"{PATH_PAIRS}/{PAIR}_{MINUTES}.csv"
TEST_INDICATORS = ["ATR", "BOLLINGER",'MA5' ,'MA25', "MA50", "MA200", "MACD", "OBV", "RSI", "STOCHASTIC", "VoRSI"]
EXPECTED_COLUMNS = [
    "v",
    "vw",
    "o",
    "c",
    "h",
    "l",
    "ATR",
    "BOLLINGER_UPPER",
    "BOLLINGER_MIDDLE",
    "BOLLINGER_LOWER",
    "MA5",
    "MA25",
    "MA50",
    "MA200",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "OBV",
    "RSI",
    "STOCHASTIC_K",
    "STOCHASTIC_D",
    "VoRSI",
]
TARGET = 'MA5'

# Preprocessor
PREPROCESSOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, "preprocessor_test"))
PREPROCESSOR_DATA_SOURCE = f"{PATH_INDICATORS}/{PAIR}_{MINUTES}.csv"
TEST_TIME_STEPS_IN = 17280 # 6 months
TEST_TIME_STEPS_OUT = 672  # 1 week
TEST_LENGTH = TEST_TIME_STEPS_IN + TEST_TIME_STEPS_OUT
TEST_SCALE = True

# Model
MODEL_DATA_SOURCE = f"{PREPROCESSOR_DATA_SOURCE}"
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_NAME = 'EURUSD_15'
TEST_EPOCHS = 500
TEST_NEURONS = 256
TEST_BATCH_SIZE = 4
TEST_LEARNING_RATE = 0.0005
PATH_TEST_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, "test_results"))