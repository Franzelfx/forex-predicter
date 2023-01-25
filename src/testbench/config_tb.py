"""Config file for testbench."""
import os.path
from datetime import datetime as date

# Data aquirer
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.curdir, 'pairs'))
PAIR = 'EURUSD'
MINUTES = 15
API_KEY = 'kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo'
TIME_FORMAT = '%Y-%m-%d'
DATE_START = '2022-10-01'
DATE_END = date.today().strftime(TIME_FORMAT)
API_TYPE = 'basic'

# Indicators
TEST_DATA_SOURCE = f"{PATH}/{PAIR}_{MINUTES}.csv"
TEST_INDICATORS = ["ATR", "BOLLINGER", "MA50", "MA200", "MACD", "RSI", "STOCHASTIC"]