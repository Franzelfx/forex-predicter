"""Module for the indicators class."""
import pandas as pd
import talib.abstract as talib
from logging import warning, info


class Indicators:
    """Used to calculate indicators from the data and add them to the dataframe.

    @remarks The indicators are calculated using the talib library.
             Please refer to https://mrjbq7.github.io/ta-lib/ for more information.
    """

    def __init__(self, data: pd.DataFrame, indicators: list):
        """Set the fundamental attributes.

        @param data: The data as a pandas dataframe with volume, volume_weighted
                        open, close, low and high as ['v', 'vw', 'o', 'c', 'l', 'h']
        @param indicators: The list of indicators to calculate.
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
        self._indicators = indicators
        self._valid_indicators = [
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
        if "MA50" in self._indicators:
            self._data_offset = 50
        elif "MA200" in self._indicators:
            self._data_offset = 200
        # Log some warning, if the indicators are not valid by comparing the lists
        if not set(self._indicators).issubset(self._valid_indicators):
            warning("One or more indicators are not valid. Please check the documentation.")

    @property
    def data(self) -> pd.DataFrame:
        """Get dataframe (could be with offset, if MA50 or MA200 are active)."""
        return self._data

    @property
    def indicators(self) -> list:
        """Get the list of indicators."""
        return self._indicators

    @property
    def valid_indicators(self) -> list:
        """Get the list of valid indicators."""
        return self._valid_indicators

    @property
    def data_offset(self) -> int:
        """Get the data offset."""
        return self._data_offset

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate the indicators and add them to the dataframe."""
        # Calculate the indicators
        if "ATR" in self._indicators:
            self._data["ATR"] = talib.ATR(
                self._data["h"],
                self._data["l"],
                self._data["close"],
                timeperiod=14,
            )
        if "BOLLINGER" in self._indicators:
            (
                self._data["bollinger_upper"],
                self._data["bollinger_middle"],
                self._data["bollinger_lower"],
            ) = talib.BBANDS(
                self._data["c"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
        if "MA50" in self._indicators:
            self._data["MA50"] = talib.MA(self._data["c"], timeperiod=50, matype=0)
        if "MA200" in self._indicators:
            self._data["MA200"] = talib.MA(
                self._data["close"], timeperiod=200, matype=0
            )
        if "MACD" in self._indicators:
            (
                self._data["macd"],
                self._data["macd_signal"],
                self._data["mcd_hist"],
            ) = talib.MACD(
                self._data["close"], fastperiod=12, slowperiod=26, signalperiod=9
            )
        if "RSI" in self._indicators:
            self._data["RSI"] = talib.RSI(self._data["c"], timeperiod=14)
        if "STOCHASTIC" in self._indicators:
            self._data["stochastik_k"], self._data["stochastik_d"] = talib.STOCH(
                self._data["h"],
                self._data["l"],
                self._data["c"],
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
        # Return the dataframe

        return self._data
