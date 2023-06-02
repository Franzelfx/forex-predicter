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
PAIR = "CADJPY"
MINUTES = 15
START = "2008-01-01"
# Substract 1 hour to get the last full hour
END = (date.today() - timedelta(hours=1)).strftime("%Y-%m-%d")
API_TYPE = "advanced"

# ---------------------------------- #
# Indicators
PATH_INDICATORS = os.path.join(currentdir, "indicators")
INDICATORS_DATA_SOURCE = f"{PATH_PAIRS}/{PAIR}_{MINUTES}.csv"
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
PREPROCESSOR_DATA_SOURCE = f"{PATH_INDICATORS}/{PAIR}_{MINUTES}.csv"
TEST_TIME_STEPS_IN = 1920  # 1 Month
TEST_TIME_STEPS_OUT = 48   # 12 hours
TEST_LENGTH = TEST_TIME_STEPS_IN + TEST_TIME_STEPS_OUT
TEST_SCALE = True
TEST_BRANCHED_MODEL = False
TEST_SHIFT = TEST_TIME_STEPS_IN # overlap of one means x and y windows are shifted by one in every sample

# Model
MODEL_DATA_SOURCE = f"{PREPROCESSOR_DATA_SOURCE}"
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_NAME = f"{PAIR}"
TEST_EPOCHS = 1000
TEST_NEURONS = 512
TEST_BATCH_SIZE = 8
TEST_LEARNING_RATE = 0.0001
TEST_PATIENCE = 300
TEST_VALIDATION_SPLIT = 0.1
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
REQUEST_PAIRS = CRYPTO_PAIRS + FOREX_PAIRS + RAW_MATERIALS
UTIL_PAIRS = CRYPTO_PAIRS + FOREX_PAIRS + RAW_MATERIALS
# UTIL_PAIRS = CRYPTO_PAIRS
TEST_BOX_PTS = 5
UTILIZER_START_DATE = START
