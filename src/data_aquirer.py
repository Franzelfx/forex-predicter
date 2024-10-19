"""Module for the DataAquirer class."""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm  # Progress bar library
from src.logger import logger as loguru

class DataAquirer:
    """Fetch data from the Polygon API or a local CSV file."""
    
    def __init__(self, path: str, api_key: str, time_format: str = "%Y-%m-%d", api_type: str = "full"):
        self.path = path
        self.api_key = api_key
        self.time_format = time_format
        self.api_type = api_type
    
    def _request(self, pair: str, time_base: int, start: str, end: str) -> pd.DataFrame:
        """Request data from API."""
        loguru.info(f"Fetching {pair} data every {time_base} minutes from {start} to {end}")
        data = pd.DataFrame()
        end = (datetime.strptime(end, self.time_format) - timedelta(days=1) 
               if self.api_type == "basic" else datetime.strptime(end, self.time_format)).strftime(self.time_format)
        last = start

        # Estimate the number of iterations based on days between start and end
        total_iterations = (datetime.strptime(end, self.time_format) - datetime.strptime(start, self.time_format)).days
        progress_bar = tqdm(total=total_iterations, desc="Fetching data", unit="day")

        while datetime.strptime(last, self.time_format) < datetime.strptime(end, self.time_format):
            url = (f"https://api.polygon.io/v2/aggs/ticker/{pair}/range/{time_base}/minute/{last}/{end}"
                   f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            response = requests.get(url).json()
            if "results" not in response:
                raise ConnectionError(response)
            
            df = pd.DataFrame(response["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            last = df["t"].iloc[-1].strftime(self.time_format)
            data = pd.concat([data, df])

            # Update the progress bar by adding the number of days fetched
            progress_bar.update(1)
        
        progress_bar.close()
        loguru.info(f"Done! Fetched {len(data)} data points.")
        return data

    def get(self, pair: str, time_base: int = 1, start: str = "2009-01-01", end: str = datetime.today().strftime("%Y-%m-%d"), 
            save: bool = False, from_file=None, no_request=False, ignore_start=False) -> pd.DataFrame:
        """Retrieve data from either a CSV file or API."""
        if from_file:
            try:
                data = pd.read_csv(f"{self.path}/{pair.split(':')[1]}_{time_base}.csv")
                loguru.info(f"Loaded data from {self.path}/{pair.split(':')[1]}_{time_base}.csv")
                recent_date = min(data["t"].iloc[-1].split()[0], (datetime.today() - timedelta(days=1)).strftime(self.time_format))
                
                if not no_request:
                    start_date = start if ignore_start else data["t"].iloc[0].split()[0]
                    request = self._request(pair, time_base, start_date, end)
                    data = pd.concat([data, request]).drop_duplicates(subset="t")
            except FileNotFoundError:
                loguru.info("Fetching data from API...")
                data = self._request(pair, time_base, start, end)
        else:
            data = self._request(pair, time_base, start, end)

        data['t'] = pd.to_datetime(data['t'])
        data.drop_duplicates(subset="t", inplace=True)
        if "C:" in pair:
            data = data[data['t'].dt.weekday < 5]  # Exclude weekends

        data = data[data["t"] < end]  # Filter data until end date
        
        if save:
            file_name = f"{self.path}/{pair.split(':')[1]}_{time_base}.csv"
            data.to_csv(file_name, index=False)
            loguru.info(f"Saved {len(data.columns)} columns to {file_name}")
        
        return data

    def remove_rows_smaller_than(self, offset: int, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove rows where a column's value is less than a specified offset."""
        return data[data[column] >= offset].reset_index(drop=True)

    def get_last_friday(self):
        """Get the most recent Friday."""
        now = datetime.now()
        return now + timedelta(days=(4 - now.weekday()) if now.weekday() <= 4 else - (now.weekday() - 4))
