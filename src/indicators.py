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
        self._path = path
        self._pair = pair
        # Check if pair name has ":" in it, if so get characters after it
        if ":" in self._pair:
            self._pair = self._pair.split(":")[1]
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
        self._data_offset = 0
        # Get data offset (cut 'MA' from the string and convert to int)
        ma = []
        if "MA5" or "MA25" or "MA50" or "MA200" in self._requested:
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
        # Calculate the indicators
        loguru.info(
            f"calculate {len(self._requested)} Indicators with {self._data.memory_usage().sum() / 1024**2:.2f} MB of data, please wait..."
        )
        if "ATR" in self._available:
            self._data["ATR"] = talib.ATR(
                self._data[keys[1]],
                self._data[keys[2]],
                self._data[keys[3]],
                timeperiod=14,
            )
        # Bollinger Bands for 15 min chart
        if "BOLLINGER" in self._requested:
            (
                self._data["BOLLINGER_UPPER"],
                self._data["BOLLINGER_MIDDLE"],
                self._data["BOLLINGER_LOWER"],
            ) = talib.BBANDS(
                self._data[bb_target], timeperiod=20, nbdevup=2.5, nbdevdn=2.5, matype=0
            )
        if "MA5" in self._requested:
            self._data["MA5"] = talib.MA(self._data[ma_target], timeperiod=5, matype=0)
        if "MA25" in self._requested:
            self._data["MA25"] = talib.MA(
                self._data[ma_target], timeperiod=25, matype=0
            )
        if "MA50" in self._requested:
            self._data["MA50"] = talib.MA(
                self._data[ma_target], timeperiod=50, matype=0
            )
        if "MA200" in self._requested:
            self._data["MA200"] = talib.MA(
                self._data[ma_target], timeperiod=200, matype=0
            )
        # Moving Average Convergence Divergence (MACD) for 15 min chart
        if "MACD" in self._requested:
            (
                self._data["MACD"],
                self._data["MACD_SIGNAL"],
                self._data["MACD_HIST"],
            ) = talib.MACD(
                self._data[macd_target], fastperiod=12, slowperiod=26, signalperiod=9
            )
        # On Balance Volume
        if "OBV" in self._requested:
            self._data["OBV"] = talib.OBV(self._data[keys[3]], self._data[keys[4]])
        # Relative Strength Index
        if "RSI" in self._requested:
            self._data["RSI"] = talib.RSI(self._data[rsi_target], timeperiod=14)
        # Stochastic Oscillator
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
        # Volume Rate of Change
        if "VoRSI" in self._requested:
            self._data["VoRSI"] = talib.RSI(self._data[vo_rsi_target], timeperiod=14)
        # Hilbert Transform - Instantaneous Trendline
        if "HT_TRENDLINE" in self._requested:
            self._data["HT_TRENDLINE"] = talib.HT_TRENDLINE(self._data[keys[3]])
        # Hilbert Transform - Trend vs Cycle Mode
        if "HT_TRENDMODE" in self._requested:
            self._data["HT_TRENDMODE"] = talib.HT_TRENDMODE(self._data[keys[3]])
        # Hilbert Transform - Dominant Cycle Period
        if "HT_DCPERIOD" in self._requested:
            self._data["HT_DCPERIOD"] = talib.HT_DCPERIOD(self._data[keys[3]])
        # Hilbert Transform - Dominant Cycle Phase
        if "HT_DCPHASE" in self._requested:
            self._data["HT_DCPHASE"] = talib.HT_DCPHASE(self._data[keys[3]])
        # Hilbert Transform - Phasor Components
        if "HT_PHASOR" in self._requested:
            (
                self._data["HT_PHASOR_INPHASE"],
                self._data["HT_PHASOR_QUADRATURE"],
            ) = talib.HT_PHASOR(self._data[keys[3]])
        # Hilbert Transform - SineWave
        if "HT_SINE" in self._requested:
            self._data["HT_SINE_SINE"], self._data["HT_SINE_LEADSINE"] = talib.HT_SINE(
                self._data[keys[3]]
            )
        if "MFI" in self._requested:
            self._data["MFI"] = talib.MFI(
                self._data[keys[1]],
                self._data[keys[2]],
                self._data[keys[3]],
                self._data[keys[4]],
                timeperiod=14,
            )
        if "MOM" in self._requested:
            self._data["MOM"] = talib.MOM(self._data[keys[3]], timeperiod=10)
        if "PLUS_DI" in self._requested:
            self._data["PLUS_DI"] = talib.PLUS_DI(
                self._data[keys[1]],
                self._data[keys[2]],
                self._data[keys[3]],
                timeperiod=14,
            )
        if "PLUS_DM" in self._requested:
            self._data["PLUS_DM"] = talib.PLUS_DM(
                self._data[keys[1]],
                self._data[keys[2]],
                timeperiod=14,
            )
        if save is True:
            path = f"{self._path}/{self._pair}_{path_extra_info}_indicators.csv"
            self._data.to_csv(path)
        # Substract the offset, if MA50 or MA200 are active
        self._data = self._data[self._data_offset :]
        return self._data

    def calculate_atr(self, data):
        return "ATR", talib.ATR(data['h'], data['l'], data['c'], timeperiod=14)

    def calculate_bollinger(self, data):
        upper, middle, lower = talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        return {"BOLLINGER_UPPER": upper, "BOLLINGER_MIDDLE": middle, "BOLLINGER_LOWER": lower}

    def calculate_ma(self, data, period):
        return f"MA{period}", talib.MA(data['c'], timeperiod=period, matype=0)

    def calculate_macd(self, data):
        macd, macdsignal, macdhist = talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)
        return {"MACD": macd, "MACD_SIGNAL": macdsignal, "MACD_HIST": macdhist}

    def calculate_obv(self, data):
        return "OBV", talib.OBV(data['c'], data['v'])

    def calculate_rsi(self, data):
        return "RSI", talib.RSI(data['c'], timeperiod=14)

    def calculate_stochastic(self, data):
        k, d = talib.STOCH(data['h'], data['l'], data['c'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        return {"STOCHASTIC_K": k, "STOCHASTIC_D": d}

    def calculate_vorsi(self, data):
        return "VoRSI", talib.RSI(data['v'], timeperiod=14)
    
    def calculate_ht_trendline(self, data):
        return "HT_TRENDLINE", talib.HT_TRENDLINE(data['c'])
    
    def calculate_ht_trendmode(self, data):
        return "HT_TRENDMODE", talib.HT_TRENDMODE(data['c'])
    
    def calculate_ht_dcperiod(self, data):
        return "HT_DCPERIOD", talib.HT_DCPERIOD(data['c'])
    
    def calculate_ht_dcphase(self, data):
        return "HT_DCPHASE", talib.HT_DCPHASE(data['c'])
    
    def calculate_ht_phasor(self, data):
        inphase, quadrature = talib.HT_PHASOR(data['c'])
        return {"HT_PHASOR_INPHASE": inphase, "HT_PHASOR_QUADRATURE": quadrature}
    
    def calculate_ht_sine(self, data):
        sine, leadsine = talib.HT_SINE(data['c'])
        return {"HT_SINE_SINE": sine, "HT_SINE_LEADSINE": leadsine}
    
    def calculate_mfi(self, data):
        return "MFI", talib.MFI(data['h'], data['l'], data['c'], data['v'], timeperiod=14)
    
    def calculate_mom(self, data):
        return "MOM", talib.MOM(data['c'], timeperiod=10)
    
    def calculate_plus_di(self, data):
        return "PLUS_DI", talib.PLUS_DI(data['h'], data['l'], data['c'], timeperiod=14)
    
    def calculate_plus_dm(self, data):
        return "PLUS_DM", talib.PLUS_DM(data['h'], data['l'], timeperiod=14)
    
    def calculate_indicators_in_parallel(self):
        indicators_functions = {
            'ATR': self.calculate_atr,
            'BOLLINGER': self.calculate_bollinger,
            'MA5': lambda data: self.calculate_ma(data, 5),
            'MA25': lambda data: self.calculate_ma(data, 25),
            'MA50': lambda data: self.calculate_ma(data, 50),
            'MA200': lambda data: self.calculate_ma(data, 200),
            'MACD': self.calculate_macd,
            'OBV': self.calculate_obv,
            'RSI': self.calculate_rsi,
            'STOCHASTIC': self.calculate_stochastic,
            'VoRSI': self.calculate_vorsi,
            'HT_TRENDLINE': self.calculate_ht_trendline,
            'HT_TRENDMODE': self.calculate_ht_trendmode,
            'HT_DCPERIOD': self.calculate_ht_dcperiod,
            'HT_DCPHASE': self.calculate_ht_dcphase,
            'HT_PHASOR': self.calculate_ht_phasor,
            'HT_SINE': self.calculate_ht_sine,
            'MFI': self.calculate_mfi,
            'MOM': self.calculate_mom,
            'PLUS_DI': self.calculate_plus_di,
            'PLUS_DM': self.calculate_plus_dm
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
