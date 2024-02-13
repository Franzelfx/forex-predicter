"""Module for the Data_Aquirer class."""
import os
import requests
import pandas as pd
from datetime import datetime as date
from datetime import datetime, timedelta
# Logging
from src.logger import logger as loguru

class Data_Aquirer:
    """Used to get the data eigther from the API or from the csv file.

    @remakrs The data aquirer is made for the Polygon API.
             please refer to https://polygon.io/docs/getting-started for more information.
    """

    def __init__(
        self, path: str, api_key: str, time_format: str = "%Y-%m-%d", api_type="full"
    ):
        """Set the attributes, may differ depending where to use it.

        @param path: The path where the fetched data should be stored.
        @param api_key: The API key for the Polygon API.
        @param time_format: The time format for the API requests and data storage.
        @param api_type: The type of the API, either 'basic' or 'premium', default is 'basic'.
        """
        self._path = path
        self._api_key = api_key
        self._time_format = time_format
        self._api_type = api_type

    @property
    def path(self) -> str:
        """Get the path."""
        return self._path

    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self._api_key

    @property
    def time_format(self) -> str:
        """Get the time format."""
        return self._time_format

    @property
    def api_type(self) -> str:
        """Get the API type."""
        return self._api_type

    @property
    def time_base(self) -> str:
        """Get the time base."""
        return f"{self._time_base} min"

    def _request(self, pair: str, time_base: int, start: str, end: str) -> pd.DataFrame:
        """Do a repeated request with the given parameters."""
        # Get data as long as the last date is not the end date of the previous day
        loguru.info(
            f"Aquiring data for {pair} with {time_base} minutes interval from {start} to {end}"
        )
        data = pd.DataFrame()
        data_return = pd.DataFrame()
        # If the api type is basic, we only have the data until yesterday
        end = datetime.strptime(end, self._time_format)
        if self._api_type == "basic":
            end = end - timedelta(days=1)
        end = datetime.strftime(end, self._time_format)
        # The last request is the start date on first iteration
        last = start
        iteration_counter = 0
        loguru.info(f"Call API ", end="", flush=True)
        while datetime.strptime(last, self._time_format) < datetime.strptime(
            end, self._time_format
        ):
            # Get the data from the API
            loguru.info(".", end="", flush=True)
            url = f"https://api.polygon.io/v2/aggs/ticker/{pair}/range/{time_base}/minute/{last}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self._api_key}"
            response = requests.get(url)
            response = response.json()
            # Check if the request was successful
            if not "results" in response:
                raise ConnectionError(response)
            # Convert the data to a pandas dataframe
            data = pd.DataFrame(response["results"])
            # Convert t from ms to datetime with given format
            data["t"] = pd.to_datetime(data["t"], unit="ms")
            data.sort_values(by="t", inplace=True)
            # Update the last date (from the last data point in the request)
            last = data["t"].iloc[-1]
            last = datetime.strftime(last, self._time_format)
            # COncatenate the data
            data_return = pd.concat([data_return, data])
            # Increment the iteration counter
            iteration_counter += 1
        # Set the time column as index
        if len(data_return) != 0:
            loguru.info(f"\nDone! (after {iteration_counter} requests).")
            loguru.info(
                f"Got {len(data_return)} data points with {data_return.memory_usage().sum() / 1024**2:.2f} MB memory usage."
            )
        else:
            loguru.info(f"\nEverything up to date.")
        return data_return

    def get(
        self,
        pair: str,
        time_base: int = 1,
        start: str = "2009-01-01",
        end: str = datetime.today().strftime("%Y-%m-%d"),
        save: bool = False,
        from_file=None,
        no_request=False,
        ignore_start=False,
    ):
        self._time_base = time_base

        # Determine the CSV file path
        csv_file_path = f"{self._path}/{pair}_{time_base}.csv"

        # Attempt to load existing data from the file
        try:
            existing_data = pd.read_csv(csv_file_path, parse_dates=["t"])
            existing_data.sort_values(by="t", inplace=True)
            last_date_in_file = existing_data["t"].max()
            loguru.info(
                f"Existing data loaded from {csv_file_path}. Last date: {last_date_in_file}"
            )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            loguru.info(f"No existing data found at {csv_file_path}. Starting fresh.")
            existing_data = None
            last_date_in_file = pd.Timestamp(start) - timedelta(
                days=1
            )  # Ensure data fetching starts from the 'start' parameter

        # Adjust the start date for new data request based on existing data
        new_data_start_date = last_date_in_file + timedelta(minutes=time_base)
        new_data_start_str = new_data_start_date.strftime("%Y-%m-%d")

        # Fetch new data only if necessary
        if not no_request and new_data_start_date.strftime("%Y-%m-%d") < end:
            new_data = self._request(pair, time_base, new_data_start_str, end)
            if existing_data is not None:
                updated_data = pd.concat([existing_data, new_data]).drop_duplicates(
                    subset=["t"], keep="last"
                )
            else:
                updated_data = new_data
        else:
            loguru.info("No new data to request. The existing dataset is up-to-date.")
            updated_data = existing_data

        if save and updated_data is not None and not updated_data.empty:
            # Save only new data by appending to the CSV file if it exists, otherwise create a new file
            if existing_data is not None and not existing_data.empty:
                new_data_only = updated_data[updated_data["t"] > last_date_in_file]
                new_data_only.to_csv(csv_file_path, mode="a", header=False, index=False)
                loguru.info(f"Appended new data to {csv_file_path}.")
            else:
                updated_data.to_csv(csv_file_path, index=False)
                loguru.info(f"Saved new data to {csv_file_path}.")

        return updated_data

    def remove_rows_smaller_than(self, offset: int, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove rows where the value of a specific column is smaller than 5.

        @param data The input DataFrame.
        @param column The name of the column to filter.
        @return The filtered DataFrame.
        """
        filtered_data = data[data[column] >= offset].reset_index(drop=True)
        return filtered_data

    def get_last_friday(self):
        now = date.now()
        closest_friday = now + timedelta(days=(4 - now.weekday()))
        return (
            closest_friday
            if closest_friday < now
            else closest_friday - timedelta(days=7)
        )
