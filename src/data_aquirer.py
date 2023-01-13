import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class Data_Aquirer():
    """Used to get the data eigther from the API or from the csv file"""

    def __init__(self, path:str, api_key: str, time_format:str):
        self.path = path
        self.api_key = api_key
        self.time_format = time_format

    def _request(self, pair: str, minutes: int, date_start: str, date_end: str):
        """Do a single request with the given parameters."""
        url = f'https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/{minutes}/minute/{date_start}/{date_end}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}'
        data = requests.get(url).json()
        if not 'results' in data:
            raise ConnectionError('No data received from the API')
        return data

    def get(self, pair: str, minutes: int, date_start: str, date_end: str, save: bool = False, from_file=False):
        """Get the data from the API or from the csv file."""
        # Check if we want to get the data from the API or from the csv file
        if from_file:
            # Get the data from the csv file
            data = pd.read_csv(f'{self.path}/{pair}_{minutes}.csv')
            # Set the time column as index
            data.set_index('t', inplace=True)
            # Return the data
            return data
        else:
            # Get the data from the API
            data = self._request(pair, minutes, date_start, date_end)
            data = pd.DataFrame(data['results'])
            data.sort_values(by='t', inplace=True)
            # Set the time column as index
            data['t'] = pd.to_datetime(data['t'], unit='ms')
            data.set_index('t', inplace=True)
            data = data[['o', 'h', 'l', 'c', 'v', 'vw']]
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volume_Weighted']
            # Add a row with the indexes from 0 to len(df)
            data['Index'] = np.arange(0, len(data))
            # Save the data to a csv file
            if save:
                data.to_csv(f'{self.path}/{pair}_{minutes}.csv')
            # Return the data
            return data
