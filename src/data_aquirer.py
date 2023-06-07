"""Module for the Data_Aquirer class."""
import os
import requests
import pandas as pd
from datetime import datetime as date
from datetime import datetime, timedelta


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

    def _request(self, pair: str, minutes: int, start: str, end: str) -> pd.DataFrame:
        """Do a repeated request with the given parameters."""
        # Get data as long as the last date is not the end date of the previous day
        print(
            f"Aquiring data for {pair} with {minutes} minutes interval from {start} to {end}"
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
        print(f"Call API ", end="", flush=True)
        while datetime.strptime(last, self._time_format) < datetime.strptime(
            end, self._time_format
        ):
            # Get the data from the API
            print(".", end="", flush=True)
            url = f"https://api.polygon.io/v2/aggs/ticker/{pair}/range/{minutes}/minute/{last}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self._api_key}"
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
            print(f"\nDone! (after {iteration_counter} requests).")
            print(
                f"Got {len(data_return)} data points with {data_return.memory_usage().sum() / 1024**2:.2f} MB memory usage."
            )
        else:
            print(f"\nEverything up to date.")
        return data_return

    def get(
        self,
        pair: str,
        minutes: int = 1,
        start: str = "2009-01-01",
        end: str = date.today().strftime("%Y-%m-%d"),
        save: bool = False,
        from_file = None,
    ):
        """Get the data from the API or from the csv file.

        @param pair: The pair to get the data for (e.g. 'EURUSD').
        @param minutes: The interval in minutes (e.g. 15min, 5min etc.).
        @param start: The start date for the data yyyy-mm-dd (e.g. 2020-01-01).
        @param end: The end date for the data yyyy-mm-dd (e.g. 2022-05-26).
        @param save: If the data should be saved to a csv file.
        @param from_file: If the data should be fetched from the csv file.

        @return: The data as pandas dataframe.
        """
        if date.today().weekday() in (6, 7):  # Check if today is weekend
            end = self.get_last_friday().strftime("%Y-%m-%d")
            print("It's weekend...")
        if from_file is not None and from_file != "" and from_file != "false":
            csv_pair_name = pair.split(":")[1] if ":" in pair else ""
            try:
                data = pd.read_csv(f"{self._path}/{csv_pair_name}_{minutes}.csv")
                print(f"Got data from {self._path}/{csv_pair_name}_{minutes}.csv")
                recent_date = data["t"].iloc[-1].split(" ")[0]
                if recent_date == date.today().strftime("%Y-%m-%d"):
                    recent_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
                request = self._request(pair, minutes, recent_date, end)
                data = pd.concat([data, request]).drop_duplicates(subset=["t"], keep='last')
            except FileNotFoundError:
                print(f"No data for {pair} with {minutes} minutes interval found.")
                print("Getting data from API...")
                data = self._request(pair, minutes, start, end)
        else:
            data = self._request(pair, minutes, start, end)
        if save:
            pair = pair.split(":")[1] if ":" in pair else pair
            print(f"Save data to {self._path}/{pair}_{minutes}.csv")
            data.to_csv(f"{self._path}/{pair}_{minutes}.csv", index=False)
            # Print column count
            print(f"Dataset has {len(data.columns)} columns.")
        return data
