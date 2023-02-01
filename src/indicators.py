"""Module for the indicators class."""
import pandas as pd
import talib.abstract as talib
from logging import warning


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
                            - 'MA50' = 'Moving Average 50'
                            - 'MA200' = 'Moving Average 200'
                            - 'MACD' = 'Moving Average Convergence Divergence'
                            - 'RSI' = 'Relative Strength Index'
                            - 'STOCHASTIC' = 'Stochastic Oscillator'
        """
        self._data = data
        self._requested = requested
        self._available = [
            "ATR",
            "BOLLINGER",
            "MA50",
            "MA200",
            "MACD",
            "RSI",
            "STOCHASTIC",
        ]
        self._data_offset = 0
        # Check, if MA50 and MA200 are in the list, if so, dataframe is
        # valid from the 50 or 200 data point.
        if "MA200" in self._requested:
            self._data_offset = 200
        elif "MA50" in self._requested:
            self._data_offset = 50
        # Log some warning, if the indicators are not valid by comparing the lists
        if not set(self._requested).issubset(self._available):
            warning("One or more indicators are not valid. Please check the documentation.")

    @property
    def data(self) -> pd.DataFrame:
        """Get dataframe (could be with offset, if MA50 or MA200 are active)."""
        return self._data[self._data_offset :]

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

    def calculate(self, save=False, path=None) -> pd.DataFrame:
        """Calculate the indicators and add them to the dataframe."""
        # Calculate the indicators
        if "ATR" in self._available:
            self._data["ATR"] = talib.ATR(
                self._data["h"],
                self._data["l"],
                self._data["c"],
                timeperiod=14,
            )
        if "BOLLINGER" in self._requested:
            (
                self._data["BOLLINGER_UPPER"],
                self._data["BOLLINGER_MIDDLE"],
                self._data["BOLLINGER_LOWER"],
            ) = talib.BBANDS(
                self._data["c"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
            )
        if "MA50" in self._requested:
            self._data["MA50"] = talib.MA(self._data["c"], timeperiod=50, matype=0)
        if "MA200" in self._requested:
            self._data["MA200"] = talib.MA(
                self._data["c"], timeperiod=200, matype=0
            )
        if "MACD" in self._requested:
            (
                self._data["MACD"],
                self._data["MACD_SIGNAL"],
                self._data["MACD_HIST"],
            ) = talib.MACD(
                self._data["c"], fastperiod=12, slowperiod=26, signalperiod=9
            )
        if "RSI" in self._requested:
            self._data["RSI"] = talib.RSI(self._data["c"], timeperiod=14)
        if "STOCHASTIC" in self._requested:
            self._data["STOCHASTIC_K"], self._data["STOCHASTIC_D"] = talib.STOCH(
                self._data["h"],
                self._data["l"],
                self._data["c"],
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
        if save and path is not None:
            self._data.to_csv(path)
        # Substract the offset, if MA50 or MA200 are active
        self._data = self._data[self._data_offset :]
        return self._data
