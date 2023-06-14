"""Config file for testbench."""
import sys
import os.path
import logging
from datetime import timedelta
from datetime import datetime as date

# ---------------------------------- #
# Add parent directory to path
currentdir = os.path.dirname(__file__)
pardir = os.path.join(currentdir, os.pardir)
sys.path.append(pardir)

# ---------------------------------- #
# Configure logging
file_path = currentdir + "/_log_/error.log"
logging.basicConfig(
    filename=file_path,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
)

# ---------------------------------- #
# Data aquirer
PATH_PAIRS = os.path.join(currentdir, "pairs")
PAIR = "C:CADJPY"
MINUTES_TRAIN = 15
MINUTES_TEST = 15
START = "2018-01-01"
# Substract 1 hour to get the last full hour
END = (date.today() - timedelta(hours=1)).strftime("%Y-%m-%d")
#END = "2023-01-01"
API_TYPE = "advanced"

# ---------------------------------- #
# Indicators
PATH_INDICATORS = os.path.join(currentdir, "indicators")
INDICATORS_DATA_SOURCE = f"{PATH_PAIRS}/{PAIR}__{MINUTES_TRAIN}.csv"
TEST_INDICATORS = [
    "ATR",
    "BOLLINGER",
    "MA5",
    "MA25",
    "MACD",
    "OBV",
    "RSI",
    "STOCHASTIC",
    "VoRSI",
    "HT_TRENDLINE",
    "HT_TRENDMODE",
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "MFI",
    "MOM",
    "PLUS_DI",
    "PLUS_DM",
]
# TEST_INDICATORS = ["BOLLINGER",'MA5' , "VoRSI"]
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
    "HT_TRENDLINE",
    "HT_TRENDMODE",
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR_INPHASE",
    "HT_PHASOR_QUADRATURE",
    "HT_SINE_SINE",
    "HT_SINE_LEADSINE",
    "MFI",
    "MOM",
    "PLUS_DI",
    "PLUS_DM",
]
TARGET = "c"

# ---------------------------------- #
# Preprocessor
PREPROCESSOR_PATH = os.path.join(currentdir, "preprocessor")
_PAIR_NAME = PAIR[2:]
PREPROCESSOR_DATA_SOURCE = f"{PATH_INDICATORS}/{_PAIR_NAME}_indicators.csv"
TEST_TIME_STEPS_IN = 960
TEST_TIME_STEPS_OUT = 96
TEST_LENGTH = TEST_TIME_STEPS_IN + TEST_TIME_STEPS_OUT
TEST_SCALE = True
TEST_BRANCHED_MODEL = False
TEST_SHIFT = TEST_TIME_STEPS_IN # overlap of one means x and y windows are shifted by one in every sample

# Model
MODEL_DATA_SOURCE = f"{PREPROCESSOR_DATA_SOURCE}"
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_NAME = f"{PAIR}"
TEST_EPOCHS = 100
TEST_NEURONS = 128
TEST_BATCH_SIZE = 1
TEST_LEARNING_RATE = 0.00005
TEST_PATIENCE = 15
TEST_VALIDATION_SPLIT = 0.2
PATH_TEST_RESULTS = os.path.join(currentdir, "test_results")

# ---------------------------------- #
# System test
FOREX_PAIRS = [
    "C:AUDCAD",
    "C:AUDCHF",
    "C:AUDJPY",
    "C:AUDNZD",
    "C:AUDUSD",
    "C:CADCHF",
    "C:CADJPY",
    "C:CHFJPY",
    "C:EURAUD",
    "C:EURCAD",
    "C:EURCHF",
    "C:EURGBP",
    "C:EURJPY",
    "C:EURNZD",
    "C:EURUSD",
    "C:GBPAUD",
    "C:GBPCAD",
    "C:GBPCHF",
    "C:GBPJPY",
    "C:GBPNZD",
    "C:GBPUSD",
    "C:NZDCAD",
    "C:NZDCHF",
    "C:NZDJPY",
    "C:NZDUSD",
    "C:USDCAD",
    "C:USDCHF",
    "C:USDJPY",
]
# FOREX_PAIRS = ['C:GBPUSD']
RAW_MATERIALS = ["C:XAGUSD"]
CRYPTO_PAIRS = [
    "X:BTCUSD",
    "X:ETHUSD",
    "X:LTCUSD",
    "X:BCHUSD",
    "X:NEOUSD",
    "X:DASHUSD",
    "X:XRPUSD",
]

# ---------------------------------- #
# Utilizer test:
REQUEST_PAIRS = FOREX_PAIRS
UTIL_PAIRS = FOREX_PAIRS
# UTIL_PAIRS = CRYPTO_PAIRS
TEST_BOX_PTS = 5
UTILIZER_START_DATE = START
