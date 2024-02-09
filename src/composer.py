"""This module composes data_aquirer, indicator, preprocessor and model into a single class."""
import os
import json
import numpy as np
import pandas as pd
from src.model import Model
from tabulate import tabulate
from src.utilizer import Utilizer
from dataclasses import dataclass
from src.model import Architecture
from src.visualizer import Visualizer
from src.indicators import Indicators
from datetime import datetime, timedelta
from src.preprocessor import Preprocessor
from src.data_aquirer import Data_Aquirer
from src.model import Branch as ModelBranch
from src.model import Output as ModelOutput
# Logger
from src.logger import logger as loguru

PATH = os.path.abspath(os.path.dirname(__file__))
PATH_RECIPES = os.path.abspath(os.path.join(PATH, "recipes"))
PATH_PAIRS = os.path.abspath(os.path.join(PATH, "pairs"))
PATH_INDICATORS = os.path.abspath(os.path.join(PATH, "indicators"))
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))


@dataclass
class Base:
    """This class is used to store the base attributes."""

    def __init__(self, json: dict):
        """Set the attributes."""
        self.api_key = json["API_KEY"]
        self.api_type = json["API_TYPE"]


@dataclass
class Processing:
    """This class is used to store the processing attributes."""

    def __init__(self, json: dict):
        """Set the attributes."""
        # Top level
        processing_sec = json["PROCESSING"]
        # Preparation
        self.pair = processing_sec["PAIR"]
        self.interval = processing_sec["INTERVAL"]
        self.start_date = processing_sec["START_DATE"]
        self.steps_in = processing_sec["STEPS_IN"]
        self.steps_out = processing_sec["STEPS_OUT"]
        self.overlap = processing_sec["OVERLAP"]
        self.test_length = processing_sec["TEST_LENGTH"]
        # Training
        self.epochs = processing_sec["EPOCHS"]
        self.batch_size = processing_sec["BATCH_SIZE"]
        self.learning_rate = processing_sec["LEARNING_RATE"]
        self.validation_split = processing_sec["VALIDATION_SPLIT"]
        self.target_feature = processing_sec["TARGET_FEATURE"]
        self.patience = processing_sec["PATIENCE"]
        self.patience_lr_schedule = processing_sec["PATIENCE_LR_SCHEDULE"]


@dataclass
class Branch:
    """This class is used to store the branching attributes."""

    def __init__(self, name: str, attributes: dict):
        """Set the attributes."""
        self.name = name
        self.indicators: list[str] = attributes["INDICATORS"]
        self.nodes: dict = attributes["NODES"]


@dataclass
class Output:
    """This class is used to store the output attributes."""

    def __init__(self, json: dict):
        """Set the attributes."""
        # Top level
        output_sec = json["OUTPUT"]
        # output
        self.nodes: dict = output_sec["NODES"]


# Alle bisherigen Importe und Klassen bleiben gleich...


@dataclass
class MainBranch:
    """This class is used to store the main branching attributes."""

    def __init__(self, attributes: dict):
        """Set the attributes."""
        self.nodes: dict = attributes["NODES"]


class Composer:
    """This class takes some recipe.json file and composes the data_aquirer, indicator, preprocessor and model into a single class."""

    def __init__(self, pair_name: str):
        """Set the fundamental attributes.

        @param The name of the pairs recipe to be used.
        """
        # Set the fundamental attributes to default values
        self._processing = None
        self._branches = {}
        self._main_branch = None
        self._output = None
        self._training = None
        self._date_time = None

        # Set the pair and model attributes
        self._pairs = None
        self._pair_names = None
        self._indicators = None
        self._preprocessed = None
        self._y_test = None
        self._y_hat = None

        # If ":" is in the pair_name, use string after ":"
        if ":" in pair_name:
            pair_name = pair_name.split(":")[1]
        recipe_path = os.path.abspath(
            os.path.join(PATH_RECIPES, f"{pair_name}_recipe.json")
        )
        base_recipe_path = os.path.abspath(
            os.path.join(PATH_RECIPES, "__BASE__recipe.json")
        )

        # Get json data from base recipe
        with open(base_recipe_path, "r") as file:
            base_recipe = json.load(file)
        # Get json data from pair specific recipe
        with open(recipe_path, "r") as file:
            recipe = json.load(file)

        # Cast the json data to dataclasses
        self._base = Base(base_recipe)
        self._processing = Processing(recipe)
        for name, attributes in recipe["BRANCHES"].items():
            self._branches[name] = Branch(name, attributes)
        self._main_branch = MainBranch(recipe["MAIN_BRANCH"])
        self._output = Output(recipe)

    @property
    def end_time(self):
        return (
            self._end_time
            if self._end_time is not None
            else datetime.today().strftime("%Y-%m-%d")
        )

    @property
    def target_candles(self):
        return self._pairs[0]

    @property
    def target_indicators(self):
        return self._branches[self._processing.pair].indicators

    @property
    def x_test(self):
        return self.target_preprocessed.x_test_target_inverse

    @property
    def y_test(self):
        return self._y_test

    @property
    def x_hat(self):
        return self.target_preprocessed.x_hat_target_inverse

    @property
    def y_hat(self):
        return self._y_hat

    @property
    def _model_branches(self) -> str:
        """Return the model tree of the attributes."""
        header = []
        table = {}
        for branch in self._branches:
            table[branch] = self._branches[branch].__dict__.copy()
            table[branch].pop("name", None)  # Remove the "name" attribute
            table[branch].pop("indicators", None)  # Remove the "indicators" attribute
            table[branch].pop("nodes", None)  # Remove the "nodes" attribute

        for branch in self._branches:
            header.append(branch)
            attr_list = []
            if "nodes" in self._branches[branch].__dict__:
                for key, value in self._branches[branch].nodes.items():
                    attr_list.append(f"{key[:2]}: {value}")
            table[branch] = attr_list

        return tabulate(table, headers=header, tablefmt="rst")

    @property
    def _model_main_branch(self) -> str:
        """Return the model tree of the attributes."""
        header = ["MAIN BRANCH"]
        table = {}
        table["MAIN BRANCH"] = self._main_branch.__dict__.copy()
        table["MAIN BRANCH"].pop("nodes", None)

        attr_list = []
        if "nodes" in self._main_branch.__dict__:
            for key, value in self._main_branch.nodes.items():
                attr_list.append(f"{key}: {value}")
        table["MAIN BRANCH"] = attr_list

        return tabulate(table, headers=header, tablefmt="rst")

    @property
    def _model_output(self) -> str:
        """Return the model tree of the attributes."""
        header = ["OUTPUT"]
        table = {}
        table["OUTPUT"] = self._output.__dict__.copy()
        table["OUTPUT"].pop("nodes", None)  # Remove the "nodes" attribute

        attr_list = []
        if "nodes" in self._output.__dict__:
            for key, value in self._output.nodes.items():
                attr_list.append(f"{key}: {value}")
        table["OUTPUT"] = attr_list

        return tabulate(table, headers=header, tablefmt="rst")

    def summary(self):
        """loguru.info the summary of the attributes."""
        loguru.info(self._model_branches)
        loguru.info(self._model_main_branch)
        loguru.info(self._model_output)

    def _synchronize_dataframes(self, dataframes, column_to_sync="t"):
        # Create a list to hold all synchronized dataframes
        synced_dataframes = []

        # Start with the first dataframe
        synced_df = dataframes[0]

        # Loop through the rest of the dataframes
        for i, df in enumerate(dataframes[1:], start=1):
            # Generate unique suffixes for each merge
            suffixes = ("_df{}".format(i), "_df{}".format(i + 1))

            # Merge on the column_to_sync with an inner join
            synced_df = pd.merge(
                synced_df, df, how="inner", on=column_to_sync, suffixes=suffixes
            )

        # Now we have a dataframe that contains rows with 't' values
        # that appear in all original dataframes. Next, we need to
        # create a synchronized version of each original dataframe.

        # Loop through the original dataframes again
        for i, df in enumerate(dataframes, start=1):
            # For each dataframe, keep only the rows that exist in synced_df
            df.columns = df.columns.str.replace(
                "_df{}".format(i), ""
            )  # Remove suffixes for final synced dataframes
            synced_dataframes.append(
                df[df[column_to_sync].isin(synced_df[column_to_sync])]
            )

        return synced_dataframes

    def _drop_time_columns(self, dataframes):
        for i, df in enumerate(dataframes):
            for column in df.columns:
                if column.startswith("t_df"):
                    df.drop(column, axis=1, inplace=True)
            dataframes[i] = df
        return dataframes

    def aquire(
        self,
        api_key: str = None,
        api_type: str = None,
        from_file=False,
        save=True,
        interval: int = None,
        no_request=False,
        ignore_start=False,
        end_time=None,
    ):
        """Aquire the data for al pairs."""
        self._end_time = end_time
        self._interval = interval if interval is not None else self._processing.interval
        if api_key is None:
            api_key = self._base.api_key
        if api_type is None:
            api_type = self._base.api_type
        # First get all pairs which are needed for the model
        # (branch names are the same as the pair names) first pair is the main pair
        pairs = []
        pair_names = []
        # pair_names.append(self._processing.pair)
        for branch in self._branches:
            pair_names.append(branch)
        for pair in pair_names:
            # Create a data_aquirer object for each pair
            aquirer = Data_Aquirer(PATH_PAIRS, api_key)
            # Recalculate the end time, if today is a weekend day (only for pair with C: in name)
            if pair.startswith("C:") and end_time is None:
                end_time = self.reconfigure_end_time(datetime.today())
            # Aquire the data
            pair = aquirer.get(
                pair,
                time_base=self._processing.interval if interval is None else interval,
                start=self._processing.start_date,
                end=end_time,
                from_file=from_file,
                save=save,
                no_request=no_request,
                ignore_start=ignore_start,
            )
            # Append the dataframes to the pairs list
            pairs.append(pair)
        # Sync the dataframes
        pairs = self._synchronize_dataframes(pairs)
        self._pairs = pairs
        self._pair_names = pair_names
        return pairs

    def calculate(self, save=False):
        """Calculate the indicators."""
        indicators = []
        for i, pair_name in enumerate(self._pair_names):
            # Create an indicator object for each pair
            # Indicator takes name of pair, dataframes of the pair, and the indicators of the pair as input
            if pair_name in self._branches:
                indicator = Indicators(
                    PATH_INDICATORS,
                    self._branches[pair_name].name,
                    self._pairs[i],
                    self._branches[pair_name].indicators,
                )
            else:
                # If pair is main pair, calculate indicators differently, as it may not have a corresponding branch in _branches.
                indicator = Indicators(
                    PATH_INDICATORS,
                    pair_name,
                    self._pairs[i],
                    self.main_pair_indicators,
                )

            # Calculate the indicators
            data = indicator.calculate(save=save)
            # Append the dataframes to the indicators list
            indicators.append(data)
        self._indicators = indicators
        return indicators

    def reconfigure_end_time(self, end_time):
        """Reconfigure the end time, if today is a weekend day."""
        # Get the current time
        now = datetime.now()
        # Get the current day
        day = now.strftime("%A")
        # If the current day is Saturday
        if day == "Saturday":
            # Subtract 1 day from the end time
            loguru.info("Today is Saturday, subtracting 1 day from end time")
            end_time = end_time - timedelta(days=1)
        # If the current day is Sunday
        elif day == "Sunday":
            # Subtract 2 days from the end time
            loguru.info("Today is Sunday, subtracting 2 days from end time")
            end_time = end_time - timedelta(days=2)
        # Return end time as string and in the correct format (same as input (end_time)
        return end_time.strftime("%Y-%m-%d")

    def preprocess(self):
        """Preprocess the data."""
        preprocessed = []
        for i, _ in enumerate(self._branches):
            # Create a preprocessor object for each pair
            preprocessor = Preprocessor(
                self._indicators[i],
                time_steps_in=self._processing.steps_in,
                time_steps_out=self._processing.steps_out,
                test_length=self._processing.test_length,
                target=self._processing.target_feature,
                overlap=self._processing.overlap,
            )
            # Append the dataframes to the preprocessed list
            preprocessed.append(preprocessor)
        self._preprocessed = preprocessed
        self.target_preprocessed = preprocessed[0]
        loguru.info(self.target_preprocessed.summary())
        return preprocessed

    def compile(self, strategy=None):
        self.model = Model(
            MODEL_PATH,
            self._processing.pair,
            self.target_preprocessed.y_train,
        )
        # Now we create our architecture
        branches = []
        i = 0
        for branch in self._branches:
            if isinstance(self._preprocessed[i].x_train, np.ndarray):
                branches.append(
                    ModelBranch(
                        self._preprocessed[i].x_train,
                        self._branches[branch].nodes["TRANSFORMER"],
                        self._branches[branch].nodes["LSTM"],
                        self._branches[branch].nodes["DENSE"],
                        self._branches[branch].nodes["ATTENTION_HEADS"],
                        self._branches[branch].nodes["DROPOUT"],
                    )
                )
            i = i + 1
        main_branch = ModelBranch(
            self.target_preprocessed.x_train,
            self._main_branch.nodes["TRANSFORMER"],
            self._main_branch.nodes["LSTM"],
            self._main_branch.nodes["DENSE"],
            self._main_branch.nodes["ATTENTION_HEADS"],
            self._main_branch.nodes["DROPOUT"],
        )
        output = ModelOutput(self._output.nodes["DENSE"], self._output.nodes["DROPOUT"])
        architecture = Architecture(branches, main_branch, output)
        self.model.compile(
            architecture,
            learning_rate=self._processing.learning_rate,
            strategy=strategy,
        )
        return self.model

    def fit(self, strategy=None, continue_training=False):
        """Fit the model."""
        self.model.fit(
            strategy=strategy,
            continue_training=continue_training,
            epochs=self._processing.epochs,
            patience=self._processing.patience,
            batch_size=self._processing.batch_size,
            validation_split=self._processing.validation_split,
            patience_lr_schedule=self._processing.patience_lr_schedule,
        )

    def predict(self, box_pts=0, test=False):
        """Predict with the model."""
        model = self.model
        self._utilizer = Utilizer(model, self._preprocessed)
        self._y_test, self._y_hat = self._utilizer.predict(box_pts=box_pts, test=test)
        visualizer = Visualizer(self._processing.pair)
        path = os.path.join(MODEL_PATH, "model_predictions")
        n = self._processing.steps_in
        m = self._processing.steps_out
        visualizer.plot_prediction(
            x_test=self.x_test,
            y_test=self._y_test,
            x_hat=self.x_hat,
            y_hat=self._y_hat,
            n=n,
            m=m,
            end_time=self.end_time,
            time_base=self._interval,
        )

    def convert_timestamp(self, item):
        loguru.info(f"Converting item: {item}")  # Debugging loguru.info statement
        if isinstance(item, pd.Timestamp):
            return item.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(item, dict):
            return {key: self.convert_timestamp(val) for key, val in item.items()}
        elif isinstance(item, list):
            return [self.convert_timestamp(elem) for elem in item]
        else:
            return item

    def dump(self, path=None, bars=2000):
        """Dump all data to a file."""
        file_path = self._prepare_file_path()
        data = self._prepare_data_to_dump(bars)

        # Check if the dump file already exists
        if os.path.isfile(file_path):
            existing_data = self._read_existing_data(file_path)
            updated_data = self._update_data_with_new_predictions(data, existing_data)
            self._write_to_file(file_path, updated_data)
        else:
            initial_data = self._prepare_initial_data(data, bars)
            self._write_to_file(file_path, initial_data)

    def _prepare_file_path(self):
        """Prepare the file path for dumping data."""
        path = os.path.join(MODEL_PATH, "model_predictions", "composer")
        if ":" in self._processing.pair:
            self._processing.pair = self._processing.pair.split(":")[1]
        return os.path.join(path, f"{self._processing.pair}_dump.json")

    def _prepare_data_to_dump(self, bars):
        """Prepare the data structure for dumping."""
        data = {
            "confidence": self._utilizer.confidence,
            "processing": {"interval": self._interval},
        }

        # Process pair data
        pair_values = self._pairs[0].tail(bars)

        # Convert 't' column to datetime if it's not already
        if not isinstance(pair_values["t"], pd.DatetimeIndex):
            pair_values["t"] = pd.to_datetime(pair_values["t"])

        # Get the last timestamp
        last_timestamp = pair_values["t"].iloc[-1]

        # Debugging: loguru.info the last timestamp
        loguru.info("Last timestamp in pair_values:", last_timestamp)
        # Prepare pair data values
        # Prepare pair data values
        data["pairs"] = {
            self._processing.pair: {
                "values": [
                    {
                        "time": row["t"].strftime("%Y-%m-%d %H:%M:%S"),
                        "open": row["o"],
                        "high": row["h"],
                        "low": row["l"],
                        "close": row["c"],
                    }
                    for index, row in pair_values.iterrows()
                ]
            }
        }

        # Prepare predictions data
        data["predictions"] = []
        for i, prediction in enumerate(self._y_hat):
            future_timestamp = last_timestamp + timedelta(
                minutes=self._interval * (i + 1)
            )
            future_timestamp_str = future_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            data["predictions"].append({"t": future_timestamp_str, "y_hat": prediction})

        return data

    def _read_existing_data(self, file_path):
        """Read existing data from the file."""
        with open(file_path, "r") as file:
            return json.load(file)
        
    def _get_common_comma_position(self, existing_data):
        comma_positions = []

        for record in existing_data:
            number_str = str(record['y_hat'])
            comma_position = number_str.find(',')
            if comma_position != -1:
                comma_positions.append(comma_position)

        if not comma_positions:
            return None

        # Find the most common comma position
        most_common_position = max(set(comma_positions), key=comma_positions.count)
        return most_common_position

    def _format_number(self, number, comma_position):
        number_str = f"{number:,}"
        if comma_position is not None and number_str.find(',') != comma_position:
            # Adjust formatting based on the most common comma position
            # This part can be complex depending on how you want to handle the formatting
            # For example, you may need to add or remove digits, or handle edge cases
            pass
        return number_str

    def _update_new_data_formatting(self, new_data, comma_position):
        for record in new_data:
            record['y_hat'] = self._format_number(record['y_hat'], comma_position)

    def _update_data_with_new_predictions(self, new_data, existing_data):
        """
        Update existing data by overriding old predictions and pair values with new ones.
        Old predictions and pair values that do not overlap with the new ones are retained.
        """
        # Update predictions
        existing_predictions = pd.DataFrame(existing_data["predictions"])
        new_predictions = pd.DataFrame(new_data["predictions"])

        # Determine the earliest timestamp in the new predictions
        earliest_new_prediction_timestamp = pd.to_datetime(new_predictions["t"]).min()

        # Filter out old predictions that overlap with new ones
        old_predictions = existing_predictions[
            pd.to_datetime(existing_predictions["t"]) < earliest_new_prediction_timestamp
        ]

        # Combine old predictions with new ones
        updated_predictions = pd.concat([old_predictions, new_predictions]).reset_index(
            drop=True
        )

        # Update predictions in the existing data
        existing_data["predictions"] = updated_predictions.to_dict(orient="records")

        # Update pair values
        existing_pair_values = pd.DataFrame(
            existing_data["pairs"][self._processing.pair]["values"]
        )
        new_pair_values = pd.DataFrame(new_data["pairs"][self._processing.pair]["values"])

        # Determine the last timestamp in the new pair values
        last_new_pair_timestamp = pd.to_datetime(new_pair_values["time"]).max()

        # Filter out old pair values that overlap with new ones
        old_pair_values = existing_pair_values[
            pd.to_datetime(existing_pair_values["time"]) <= last_new_pair_timestamp
        ]

        # Get the most common comma position from existing data
        common_comma_position = self._get_common_comma_position(existing_data["predictions"])

        # Format new data accordingly
        self._update_new_data_formatting(new_data["predictions"], common_comma_position)

        # Combine old pair values with new ones
        updated_pair_values = pd.concat([old_pair_values, new_pair_values]).reset_index(
            drop=True
        )

        # Update pair values in the existing data
        existing_data["pairs"][self._processing.pair][
            "values"
        ] = updated_pair_values.to_dict(orient="records")

        return existing_data

    def _prepare_initial_data(self, data, bars):
        """Prepare initial data if dump file does not exist."""
        timestamps = self._pairs[0]["t"].tail(bars).tolist()
        datetime_objects = [ts.to_pydatetime() for ts in timestamps]
        last_timestamp = datetime_objects[-1]

        loguru.info("Last timestamp in initial data:", last_timestamp)

        data["predictions"] = []
        for i in range(len(self._y_hat)):
            # Convert the interval from minutes to a timedelta
            interval_timedelta = timedelta(minutes=self._interval * (i + 1))

            # Add the timedelta to the last_timestamp
            future_timestamp = last_timestamp + interval_timedelta
            future_timestamp_str = future_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            prediction = self._y_hat[i]

            # Add each prediction as a JSON object
            data["predictions"].append({"t": future_timestamp_str, "y_hat": prediction})

        return data

    def _write_to_file(self, file_path, data):
        """Write data to a JSON file."""

        def default_converter(o):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(o, np.float32):
                return float(o)
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4, default=default_converter)
