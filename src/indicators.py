"""Module for the indicators class."""
import os
import pandas as pd
import concurrent.futures
from logging import warning
import talib.abstract as talib

# Logging
from src.logger import logger as loguru

class Indicators:
    def __init__(self, path: str, pair: str, data: pd.DataFrame, requested: list):
        """Set the fundamental attributes."""
        self._path = path
        self._pair = pair
        if ":" in self._pair:
            self._pair = self._pair.split(":")[1]
        
        # Use the original column names
        self._data = data
        self._requested = requested
        self._available = [
            "ATR", "BOLLINGER", "RSI", "MACD", "MOM", "MA5", "MA25", "MA50", "MA200",
            "ADX", "SAR", "OBV", "MFI", "HT_TRENDMODE", "HT_SINE"
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
            warning("One or more indicators are not valid. Please check the documentation.")

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

    def calculate_atr(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the ATR indicator."""
        return {"ATR": talib.ATR(data[keys[1]], data[keys[2]], data[keys[3]], timeperiod=14)}

    def calculate_bollinger(self, data: pd.DataFrame, bb_target="c") -> dict:
        """Calculate the Bollinger Bands."""
        upper, middle, lower = talib.BBANDS(data[bb_target], timeperiod=20, nbdevup=2.5, nbdevdn=2.5, matype=0)
        return {
            "BOLLINGER_UPPER": upper,
            "BOLLINGER_MIDDLE": middle,
            "BOLLINGER_LOWER": lower
        }

    def calculate_rsi(self, data: pd.DataFrame, rsi_target) -> dict:
        """Calculate the RSI indicator."""
        return {"RSI": talib.RSI(data[rsi_target], timeperiod=14)}

    def calculate_macd(self, data: pd.DataFrame, macd_target) -> dict:
        """Calculate the MACD indicator."""
        macd, macd_signal, macd_hist = talib.MACD(
            data[macd_target], fastperiod=12, slowperiod=26, signalperiod=9
        )
        return {
            "MACD": macd,
            "MACD_SIGNAL": macd_signal,
            "MACD_HIST": macd_hist
        }

    def calculate_mom(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the Momentum indicator."""
        return {"MOM": talib.MOM(data[keys[3]], timeperiod=10)}

    def calculate_ma(self, data: pd.DataFrame, ma_target: str, period: int) -> dict:
        """Calculate the Moving Average for the given period."""
        return {f"MA{period}": talib.MA(data[ma_target], timeperiod=period, matype=0)}

    def calculate_adx(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the ADX indicator."""
        return {"ADX": talib.ADX(data[keys[1]], data[keys[2]], data[keys[3]], timeperiod=14)}

    def calculate_sar(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the SAR indicator."""
        return {"SAR": talib.SAR(data[keys[1]], data[keys[2]], acceleration=0.02, maximum=0.2)}

    def calculate_obv(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the OBV indicator."""
        return {"OBV": talib.OBV(data[keys[3]], data[keys[4]])}

    def calculate_mfi(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the MFI indicator."""
        return {"MFI": talib.MFI(data[keys[1]], data[keys[2]], data[keys[3]], data[keys[4]], timeperiod=14)}

    def calculate_ht_trendmode(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the HT Trend Mode indicator."""
        return {"HT_TRENDMODE": talib.HT_TRENDMODE(data[keys[3]])}

    def calculate_ht_sine(self, data: pd.DataFrame, keys) -> dict:
        """Calculate the HT Sine indicator."""
        sine, leadsine = talib.HT_SINE(data[keys[3]])
        return {"HT_SINE_SINE": sine, "HT_SINE_LEADSINE": leadsine}

    def calculate_indicators_in_parallel(self):
        # Map the short column names to the expected full column names for indicator calculation
        column_mapping = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        
        # Rename columns to full names
        self._data = self._data.rename(columns=column_mapping)
        
        indicators_functions = {
            'ATR': self.calculate_atr,
            'BOLLINGER': self.calculate_bollinger,
            'RSI': self.calculate_rsi,
            'MACD': self.calculate_macd,
            'MOM': self.calculate_mom,
            'MA5': lambda data, keys: self.calculate_ma(data, 'close', 5),
            'MA25': lambda data, keys: self.calculate_ma(data, 'close', 25),
            'MA50': lambda data, keys: self.calculate_ma(data, 'close', 50),
            'MA200': lambda data, keys: self.calculate_ma(data, 'close', 200),
            'ADX': self.calculate_adx,
            'SAR': self.calculate_sar,
            'OBV': self.calculate_obv,
            'MFI': self.calculate_mfi,
            'HT_TRENDMODE': self.calculate_ht_trendmode,
            'HT_SINE': self.calculate_ht_sine
        }

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self._data.columns]

        if missing_columns:
            loguru.warning(f"Data is missing required columns: {missing_columns}")
            return self._data  # Return early if required columns are missing
        
        # Loguru.info the indicators that will be calculated
        loguru.info(f"Calculating the following indicators: {self._requested}")
        
        # Limit the number of threads to avoid overloading the system
        max_threads = len(os.sched_getaffinity(0))

        # Calculate indicators in parallel
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            futures = {
                indicator: executor.submit(func, self._data, ['open', 'high', 'low', 'close', 'volume']) 
                for indicator, func in indicators_functions.items() 
                if indicator in self._requested
            }

            for indicator, future in futures.items():
                try:
                    result = future.result()
                    if isinstance(result, dict):
                        for key, value in result.items():
                            self._data[key] = value
                except Exception as exc:
                    loguru.warning(f'{indicator} generated an exception: {exc}')

        # After calculations, map columns back to the original short names
        reverse_column_mapping = {v: k for k, v in column_mapping.items()}
        self._data = self._data.rename(columns=reverse_column_mapping)

        return self._data

