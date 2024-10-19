import os
import asyncio
import aiohttp
import aiofiles
import pandas as pd
from io import StringIO
from tqdm.asyncio import tqdm  # Async progress bar library
from src.logger import logger as loguru
from datetime import datetime, timedelta

class DataAquirer:
    """Fetch data from the Polygon API or a local CSV file asynchronously."""
    
    def __init__(self, path: str, api_key: str, time_format: str = "%Y-%m-%d", api_type: str = "full"):
        self.path = path
        self.api_key = api_key
        self.time_format = time_format
        self.api_type = api_type

    async def _make_request(self, session: aiohttp.ClientSession, pair: str, time_base: int, start: str, end: str) -> pd.DataFrame:
        """Make a single request to the API asynchronously."""
        url = (f"https://api.polygon.io/v2/aggs/ticker/{pair}/range/{time_base}/minute/{start}/{end}"
               f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
        async with session.get(url) as response:
            response.raise_for_status()  # Raise exception if request failed
            data = await response.json()
            if "results" not in data:
                raise ValueError(f"No results in the response for {pair} between {start} and {end}")
            df = pd.DataFrame(data["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            return df

    async def _request_parallel(self, pair: str, time_base: int, start: str, end: str) -> pd.DataFrame:
        """Fetch data from the API using multiple asynchronous requests."""
        loguru.info(f"Fetching {pair} data every {time_base} minutes from {start} to {end}")
        
        start_date = datetime.strptime(start, self.time_format)
        end_date = datetime.strptime(end, self.time_format)
        delta = timedelta(days=30)  # Request 30 days of data in each API call

        # Split time range into intervals
        date_ranges = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + delta, end_date)
            date_ranges.append((current_start.strftime(self.time_format), current_end.strftime(self.time_format)))
            current_start = current_end

        # Asynchronously fetch data for each interval
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._make_request(session, pair, time_base, s, e) for s, e in date_ranges
            ]
            data_frames = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Fetching {pair} data", unit="request"):
                try:
                    data_frames.append(await task)
                except Exception as e:
                    loguru.error(f"Failed to fetch data for {pair}: {e}")

        # Combine all DataFrames into one
        if data_frames:
            data = pd.concat(data_frames, ignore_index=True).drop_duplicates(subset="t")
            return data
        else:
            loguru.warning(f"No data fetched for {pair}")
            return pd.DataFrame()

    async def async_read_csv(self, file_path: str) -> pd.DataFrame:
        """Asynchronously read a CSV file into a pandas DataFrame."""
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
        return pd.read_csv(StringIO(content))

    async def async_write_csv(self, file_path: str, data: pd.DataFrame):
        """Asynchronously write a pandas DataFrame to a CSV file."""
        async with aiofiles.open(file_path, mode='w') as f:
            await f.write(data.to_csv(index=False))

    async def get(self, pair: str, time_base: int = 1, start: str = "2009-01-01", end: str = datetime.today().strftime("%Y-%m-%d"), 
                  save: bool = False, from_file: bool = False, no_request: bool = False, ignore_start: bool = False) -> pd.DataFrame:
        """Retrieve data from either a CSV file or API asynchronously."""
        file_path = f"{self.path}/{pair.split(':')[1]}_{time_base}.csv"
        data = pd.DataFrame()

        if from_file:
            try:
                data = await self.async_read_csv(file_path)
                loguru.info(f"Loaded data from {file_path}")
                recent_date = min(data["t"].iloc[-1].split()[0], (datetime.today() - timedelta(days=1)).strftime(self.time_format))
                
                if not no_request:
                    start_date = start if ignore_start else data["t"].iloc[0].split()[0]
                    additional_data = await self._request_parallel(pair, time_base, start_date, end)
                    data = pd.concat([data, additional_data]).drop_duplicates(subset="t")
            except FileNotFoundError:
                loguru.info("File not found, fetching data from API...")
                data = await self._request_parallel(pair, time_base, start, end)
        else:
            data = await self._request_parallel(pair, time_base, start, end)

        data['t'] = pd.to_datetime(data['t'])
        data = data.drop_duplicates(subset="t")
        if "C:" in pair:
            data = data[data['t'].dt.weekday < 5]  # Exclude weekends

        data = data[data["t"] < end]  # Filter data until end date
        
        if save:
            await self.async_write_csv(file_path, data)
            loguru.info(f"Saved {len(data.columns)} columns to {file_path}")
        
        return data

    async def remove_rows_smaller_than(self, offset: int, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove rows where a column's value is less than a specified offset asynchronously."""
        return data[data[column] >= offset].reset_index(drop=True)

    async def get_last_friday(self) -> datetime:
        """Get the most recent Friday asynchronously."""
        now = datetime.now()
        return now + timedelta(days=(4 - now.weekday()) if now.weekday() <= 4 else - (now.weekday() - 4))
