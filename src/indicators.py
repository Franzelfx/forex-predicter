"""Module for the indicators class."""
import pandas as pd
from logging import warning
from tabulate import tabulate
import talib.abstract as talib


class Indicators:
    """Used to calculate indicators from the data and add them to the dataframe.

    @remarks The indicators are calculated using the talib library.
             Please refer to https://mrjbq7.github.io/ta-lib/ for more information.
    """

    def __init__(self, data: pd.DataFrame, requested: list):
        """Set the fundamental attributes.

        @param data: The data as a pandas dataframe with volume, volume_weighted
                        open, close, low and high as ['v', 'vw', 'o', 'c', 'l', 'h']
        @param requested: The list of indicators to calculate.
                           Available Indicators are:

                            - 'ATR' = 'Average True Range'
                            - 'BOLLINGER' = 'Bollinger Bands'
                            - 'MA5' = 'Moving Average 5'
                            - 'MA25' = 'Moving Average 25'
                            - 'MA50' = 'Moving Average 50'
                            - 'MA200' = 'Moving Average 200'
                            - 'MACD' = 'Moving Average Convergence Divergence'
                            - 'OBV' = 'On Balance Volume'
                            - 'RSI' = 'Relative Strength Index'
                            - 'STOCHASTIC' = 'Stochastic Oscillator'
                            - 'VoRSI' = 'Volume Relative Strength Index'
        """
        self._data = data
        self._requested = requested
        self._available = [
            "ATR",
            "BOLLINGER",
            "MA5",
            "MA25",
            "MA50",
            "MA200",
            "MACD",
            "OBV",
            "RSI",
            "STOCHASTIC",
            "VoRSI",
        ]
        self._data_offset = 0
        # Get data offset (cut 'MA' from the string and convert to int)
        ma = []
        if 'MA5' or 'MA25' or 'MA50' or 'MA200' in self._requested:
            # Get the maximum of the moving averages
            for i in self._requested:
                if i.startswith("MA"):
                    # Chek, if the string is a number
                    try:
                        ma.append(int(i[2:]))
                    except ValueError:
                        pass
            self._data_offset = max(ma)
        # Log some warning, if the indicators are not valid by comparing the lists
        if not set(self._requested).issubset(self._available):
            warning("One or more indicators are not valid. Please check the documentation.")

    def summary(self):
        """Get a summary of the indicators."""
        print(self._data.head(5))

    @property
    def data(self) -> pd.DataFrame:
        """Get dataframe (could be with offset, if MA50 or MA200 are active)."""
        return self._data[self._data_offset:]

    @property
    def requested(self) -> list:
        """Get the list of indicators."""
        return self._requested

    @property
    def available(self) -> list:
        """Get the list of valid indicators."""
        return self._available

    @property
    def data_offset(self) -> int:
        """Get the data offset."""
        return self._data_offset

    def calculate(self, save=False, path=None, ma_target='c', keys=['o', 'h', 'l', 'c', 'v'], macd_target='c', bb_target='c', rsi_target='c', vo_rsi_target='v') -> pd.DataFrame:
        """Calculate the indicators and add them to the dataframe."""
        # Calculate the indicators
        print(f"calculate {len(self._requested)} Indicators for {self._data.memory_usage().sum() / 1024**2:.2f} MB of data.")
        if "ATR" in self._available:
            self._data["ATR"] = talib.ATR(
                self._data[keys[1]],
                self._data[keys[2]],
                self._data[keys[3]],
                timeperiod=14,
            )
        if "BOLLINGER" in self._requested:
            (
                self._data["BOLLINGER_UPPER"],
                self._data["BOLLINGER_MIDDLE"],
                self._data["BOLLINGER_LOWER"],
            ) = talib.BBANDS(
                self._data[bb_target], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
            )
        if "MA5" in self._requested:
            self._data["MA5"] = talib.MA(self._data[ma_target], timeperiod=5, matype=0)
        if "MA25" in self._requested:
            self._data["MA25"] = talib.MA(self._data[ma_target], timeperiod=25, matype=0)
        if "MA50" in self._requested:
            self._data["MA50"] = talib.MA(self._data[ma_target], timeperiod=50, matype=0)
        if "MA200" in self._requested:
            self._data["MA200"] = talib.MA(
                self._data[ma_target], timeperiod=200, matype=0
            )
        if "MACD" in self._requested:
            (
                self._data["MACD"],
                self._data["MACD_SIGNAL"],
                self._data["MACD_HIST"],
            ) = talib.MACD(
                self._data[macd_target], fastperiod=12, slowperiod=26, signalperiod=9
            )
        if "OBV" in self._requested:
            self._data["OBV"] = talib.OBV(self._data[keys[3]], self._data[keys[4]])
        if "RSI" in self._requested:
            self._data["RSI"] = talib.RSI(self._data[rsi_target], timeperiod=14)
        if "STOCHASTIC" in self._requested:
            self._data["STOCHASTIC_K"], self._data["STOCHASTIC_D"] = talib.STOCH(
                self._data[keys[1]],
                self._data[keys[2]],
                self._data[keys[3]],
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
        if "VoRSI" in self._requested:
            self._data["VoRSI"] = talib.RSI(self._data[vo_rsi_target], timeperiod=14)
        if save and path is not None:
            self._data.to_csv(path)
        # Substract the offset, if MA50 or MA200 are active
        self._data = self._data[self._data_offset:]
        return self._data
