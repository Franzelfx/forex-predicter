"""Module for the indicators class."""
import pandas as pd
import concurrent.futures
from logging import warning
import talib.abstract as talib

# Logging
from src.logger import logger as loguru

class Indicators:
    """Used to calculate indicators from the data and add them to the dataframe.

    @remarks The indicators are calculated using the talib library.
             Please refer to https://mrjbq7.github.io/ta-lib/ for more information.
    """

    def __init__(self, path: str, pair: str, data: pd.DataFrame, requested: list):
        """Set the fundamental attributes."""
        self._path = path
        self._pair = pair
        if ":" in self._pair:
            self._pair = self._pair.split(":")[1]
        self._data = data
        self._requested = requested
        self._available = [
            "ATR",
            "BOLLINGER",
            "RSI",
            "MACD",
            "MOM",
            "MA5",
            "MA25",
            "MA50",
            "MA200",
            "ADX",
            "SAR",
            "OBV",
            "MFI",
            "HT_TRENDMODE",
            "HT_SINE"
        ]
        self._data_offset = 0

        ma = []
        if "MA5" or "MA25" or "MA50" or "MA200" in self._requested:
            for i in self._requested:
                if i.startswith("MA"):
                    try:
                        ma.append(int(i[2:]))
                    except ValueError:
                        pass
            self._data_offset = max(ma)
        if not set(self._requested).issubset(self._available):
            warning(
                "One or more indicators are not valid. Please check the documentation."
            )

    def summary(self):
        """loguru.info a summary of the indicators."""
        loguru.info(self._data.head(5))

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

    def calculate(
        self,
        save=False,
        path=None,
        ma_target="c",
        keys=["o", "h", "l", "c", "v"],
        macd_target="c",
        bb_target="c",
        rsi_target="c",
        vo_rsi_target="v",
        path_extra_info="",
    ) -> pd.DataFrame:
        """Calculate the indicators and add them to the dataframe."""
        loguru.info(
            f"calculate {len(self._requested)} Indicators with {self._data.memory_usage().sum() / 1024**2:.2f} MB of data, please wait..."
        )
        if "ATR" in self._requested:
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
                self._data[bb_target], timeperiod=20, nbdevup=2.5, nbdevdn=2.5, matype=0
            )
        if "RSI" in self._requested:
            self._data["RSI"] = talib.RSI(self._data[rsi_target], timeperiod=14)
        if "MACD" in self._requested:
            (
                self._data["MACD"],
                self._data["MACD_SIGNAL"],
                self._data["MACD_HIST"],
            ) = talib.MACD(
                self._data[macd_target], fastperiod=12, slowperiod=26, signalperiod=9
            )
        if "MOM" in self._requested:
            self._data["MOM"] = talib.MOM(self._data[keys[3]], timeperiod=10)
        if "MA5" in self._requested:
            self._data["MA5"] = talib.MA(self._data[ma_target], timeperiod=5, matype=0)
        if "MA25" in self._requested:
            self._data["MA25"] = talib.MA(self._data[ma_target], timeperiod=25, matype=0)
        if "MA50" in self._requested:
            self._data["MA50"] = talib.MA(self._data[ma_target], timeperiod=50, matype=0)
        if "MA200" in self._requested:
            self._data["MA200"] = talib.MA(self._data[ma_target], timeperiod=200, matype=0)
        if "ADX" in self._requested:
            self._data["ADX"] = talib.ADX(self._data[keys[1]], self._data[keys[2]], self._data[keys[3]], timeperiod=14)
        if "SAR" in self._requested:
            self._data["SAR"] = talib.SAR(self._data[keys[1]], self._data[keys[2]], acceleration=0.02, maximum=0.2)
        if "OBV" in self._requested:
            self._data["OBV"] = talib.OBV(self._data[keys[3]], self._data[keys[4]])
        if "MFI" in self._requested:
            self._data["MFI"] = talib.MFI(
                self._data[keys[1]],
                self._data[keys[2]],
                self._data[keys[3]],
                self._data[keys[4]],
                timeperiod=14,
            )
        if "HT_TRENDMODE" in self._requested:
            self._data["HT_TRENDMODE"] = talib.HT_TRENDMODE(self._data[keys[3]])
        if "HT_SINE" in self._requested:
            self._data["HT_SINE_SINE"], self._data["HT_SINE_LEADSINE"] = talib.HT_SINE(
                self._data[keys[3]]
            )

        if save is True:
            path = f"{self._path}/{self._pair}_{path_extra_info}_indicators.csv"
            self._data.to_csv(path)
        
        self._data = self._data[self._data_offset :]
        return self._data

    def calculate_indicators_in_parallel(self):
        indicators_functions = {
            'ATR': self.calculate_atr,
            'BOLLINGER': self.calculate_bollinger,
            'RSI': self.calculate_rsi,
            'MACD': self.calculate_macd,
            'MOM': self.calculate_mom,
            'MA5': lambda data: self.calculate_ma(data, 5),
            'MA25': lambda data: self.calculate_ma(data, 25),
            'MA50': lambda data: self.calculate_ma(data, 50),
            'MA200': lambda data: self.calculate_ma(data, 200),
            'ADX': self.calculate_adx,
            'SAR': self.calculate_sar,
            'OBV': self.calculate_obv,
            'MFI': self.calculate_mfi,
            'HT_TRENDMODE': self.calculate_ht_trendmode,
            'HT_SINE': self.calculate_ht_sine
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for indicator_name in self._requested:
                func = indicators_functions.get(indicator_name)
                if func:
                    futures.append(executor.submit(func, self._data))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, dict):
                    self._data.update(result)
                else:
                    self._data[result[0]] = result[1]

        return self._data
