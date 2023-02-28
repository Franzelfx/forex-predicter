"""Config file for testbench."""
import sys
import os.path
from api_key import API_KEY
from datetime import datetime as date


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Data aquirer
PATH_PAIRS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, "pairs"))
PAIR = "EURUSD"
MINUTES = 1
START = "2010-01-01"
API_TYPE = "advanced"
API_KEY = API_KEY

# Indicators
PATH_INDICATORS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, f"indicators"))
INDICATORS_DATA_SOURCE = f"{PATH_PAIRS}/{PAIR}_{MINUTES}.csv"
TEST_INDICATORS = ["ATR", "BOLLINGER",'MA5' ,'MA25', "MA50", "MA200", "MACD", "OBV", "RSI", "STOCHASTIC", "VoRSI"]
#TEST_INDICATORS = ["BOLLINGER",'MA5' ,'MA25', "MA50", "MA200", "VoRSI"]
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
TEST_TIME_STEPS_IN = 2160  # 18 hours
TEST_TIME_STEPS_OUT = 720  # 12 hours
TEST_LENGTH = TEST_TIME_STEPS_IN + TEST_TIME_STEPS_OUT
TEST_SCALE = True
TEST_BRANCHED_MODEL = False
TEST_SHIFT = 1440 # overlap of one means x and y wndows are shifted by one in every sample

# Model
MODEL_DATA_SOURCE = f"{PREPROCESSOR_DATA_SOURCE}"
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_NAME = f"{PAIR}_{MINUTES}"
TEST_EPOCHS = 300
TEST_NEURONS = 256
TEST_BATCH_SIZE = 16
TEST_LEARNING_RATE = 0.0004
TEST_VALIDATION_SPLIT = 0.4
PATH_TEST_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, "test_results"))

# System test
REQUEST_PAIRS = ['AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
# Utilizer test
UTIL_PAIRS = ['AUDUSD', 'CADCHF', 'CADJPY']
#UTIL_PAIRS = ['GBPJPY', 'EURUSD']
