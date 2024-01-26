import yfinance as yf
import pandas as pd
from datetime import datetime

class ForexCorrelationCalculator:
    def __init__(self, forex_pair_1, forex_pair_2, start_date=datetime(2020, 1, 1), end_date=datetime.now()):
        self.forex_pair_1 = forex_pair_1 + "=X"
        self.forex_pair_2 = forex_pair_2 + "=X"
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self, forex_pair):
        try:
            data = yf.download(forex_pair, start=self.start_date, end=self.end_date)
            return data['Adj Close'].pct_change().dropna()
        except Exception as e:
            print(f"Error fetching data for {forex_pair}: {e}")
            return pd.Series()

    def calculate_correlation(self):
        returns_1 = self.get_data(self.forex_pair_1)
        returns_2 = self.get_data(self.forex_pair_2)

        if not returns_1.empty and not returns_2.empty:
            correlation = returns_1.corr(returns_2)
            print(f"The correlation between {self.forex_pair_1} and {self.forex_pair_2} is: {correlation}")
        else:
            print("Unable to calculate correlation due to insufficient data.")

if __name__ == '__main__':
    calculator = ForexCorrelationCalculator('NZDUSD', 'AUDUSD')
    calculator.calculate_correlation()