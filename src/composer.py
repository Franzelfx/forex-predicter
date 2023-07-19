"""This module composes data_aquirer, indicator, preprocessor and model into a single class."""
import os
import json
import pandas as pd
import nupy as np
from src.data_aquirer import Data_Aquirer
from src.indicators import Indicators
from src.preprocessor import Preprocessor
from src.model import Model
from tabulate import tabulate
from dataclasses import dataclass
from src.model import Architecture, Branch, Output

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
        self.steps_in = processing_sec["STEPS_IN"]
        self.steps_out = processing_sec["STEPS_OUT"]
        self.test_length = processing_sec["TEST_LENGTH"]
        # Training
        self.epochs = processing_sec["EPOCHS"]
        self.batch_size = processing_sec["BATCH_SIZE"]
        self.learning_rate = processing_sec["LEARNING_RATE"]
        self.validation_split = processing_sec["VALIDATION_SPLIT"]
        self.target_feature = processing_sec["TARGET_FEATURE"]
        self.patience = processing_sec["PATIENCE"]
        self.patience_lr_chedule = processing_sec["PATIENCE_LR_SCHEDULE"]


@dataclass
class Branch:
    """This class is used to store the branching attributes."""

    def __init__(self, name: str, attributes: dict):
        """Set the attributes."""
        self.name = name
        self.indicators: list[str] = attributes["INDICATORS"]
        self.nodes: list[dict] = attributes["NODES"]


@dataclass
class Output:
    """This class is used to store the output attributes."""

    def __init__(self, json: dict):
        """Set the attributes."""
        # Top level
        output_sec = json["OUTPUT"]
        # output
        self.nodes: list[dict] = output_sec["NODES"]


# Alle bisherigen Importe und Klassen bleiben gleich...


@dataclass
class MainBranch:
    """This class is used to store the main branching attributes."""

    def __init__(self, attributes: dict):
        """Set the attributes."""
        self.nodes: list[dict] = attributes["NODES"]


class Composer:
    """This class takes some recipe.json file and composes the data_aquirer, indicator, preprocessor and model into a single class."""

    def __init__(self, pair_name: str):
        """Set the fundamental attributes.

        @param recipe: Path to the recipe.json file.
        """
        # Set the fundamental attributes to default values
        self._processing = None
        self._branches = {}
        self._main_branch = None
        self._output = None
        self._training = None

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
                for key, value in self._branches[branch].nodes[0].items():
                    attr_list.append(f"{key}: {value}")
            table[branch] = attr_list

        return tabulate(table, headers=header, tablefmt="rst")

    def _model_main_branch(self) -> str:
        """Return the model tree of the attributes."""
        header = ["MAIN BRANCH"]
        table = {}
        table["MAIN BRANCH"] = self._main_branch.__dict__.copy()
        table["MAIN BRANCH"].pop("nodes", None)

        attr_list = []
        if "nodes" in self._main_branch.__dict__:
            for key, value in self._main_branch.nodes[0].items():
                attr_list.append(f"{key}: {value}")
        table["MAIN BRANCH"] = attr_list

        return tabulate(table, headers=header, tablefmt="rst")

    def _model_output(self) -> str:
        """Return the model tree of the attributes."""
        header = ["OUTPUT"]
        table = {}
        table["OUTPUT"] = self._output.__dict__.copy()
        table["OUTPUT"].pop("nodes", None)  # Remove the "nodes" attribute

        attr_list = []
        if "nodes" in self._output.__dict__:
            for key, value in self._output.nodes[0].items():
                attr_list.append(f"{key}: {value}")
        table["OUTPUT"] = attr_list

        return tabulate(table, headers=header, tablefmt="rst")

    def summary(self):
        """Print the summary of the attributes."""
        print(self._model_branches())
        print(self._model_main_branch())
        print(self._model_output())

    def _synchronize_dataframes(dataframes, column_to_sync="t"):
        # Create a list to hold all synchronized dataframes
        synced_dataframes = []

        # Start with the first dataframe
        synced_df = dataframes[0]

        # Loop through the rest of the dataframes
        for df in dataframes[1:]:
            # Merge on the column_to_sync with an inner join
            synced_df = pd.merge(synced_df, df, how="inner", on=column_to_sync)

        # Now we have a dataframe that contains rows with 't' values
        # that appear in all original dataframes. Next, we need to
        # create a synchronized version of each original dataframe.

        # Loop through the original dataframes again
        for df in dataframes:
            # For each dataframe, keep only the rows that exist in synced_df
            synced_dataframes.append(
                df[df[column_to_sync].isin(synced_df[column_to_sync])]
            )
        return synced_dataframes

    def aquire(self, api_key: str = None, api_type: str = None):
        """Aquire the data for al pairs."""
        if api_key is None:
            api_key = self._base.api_key
        if api_type is None:
            api_type = self._base.api_type
        # First get all pairs which are needed for the model
        # (branch names are the same as the pair names) first pair is the main pair
        pairs = []
        pair_names = []
        pair_names.append(self._processing.pair)
        for branch in self._branches:
            pair_names.append(branch)
        for pair in pair_names:
            # Create a data_aquirer object for each pair
            aquirer = Data_Aquirer(pair, api_key, api_type)
            # Aquire the data
            aquirer.aquire(
                interval=self._processing.interval,
                steps_in=self._processing.steps_in,
                steps_out=self._processing.steps_out,
                test_length=self._processing.test_length,
            )
            # Append the dataframes to the pairs list
            pairs.append(aquirer.dataframes)
        # Sync the dataframes
        pairs = self._synchronize_dataframes(pairs)
        self.pairs = pairs
        self.pair_names = pair_names
        return pairs

    def calculate(self, save=False):
        """Calculate the indicators."""
        indicators = []
        for i, pair_name in enumerate(self.pair_names):
            # Create an indicator object for each pair
            # Indicator takes name of pair, dataframes of the pair, and the indicators of the pair as input
            if pair_name in self._branches:
                indicator = Indicators(
                    PATH_INDICATORS,
                    self._branches[pair_name],
                    self.pairs[i],
                    self._branches.indicators,
                )
            else:
                # If pair is main pair, calculate indicators differently, as it may not have a corresponding branch in _branches.
                indicator = Indicators(
                    PATH_INDICATORS, pair_name, self.pairs[i], self.main_pair_indicators
                )

            # Calculate the indicators
            indicator.calculate(save=save)
            # Append the dataframes to the indicators list
            indicators.append(indicator.dataframes)
        self.indicators = indicators
        return indicators

    def preprocess(self):
        """Preprocess the data."""
        preprocessed = []
        for i in enumerate(self.pair_names):
            # Create a preprocessor object for each pair
            preprocessor = Preprocessor(
                self.indicators[i],
                time_steps_in=self._processing.steps_in,
                time_steps_out=self._processing.steps_out,
                test_length=self._processing.test_length,
                target=self._processing.target_feature,
            )
            # Append the dataframes to the preprocessed list
            preprocessed.append(preprocessor.dataframes)
        self.preprocessed = preprocessed
        self.target_preprocessed = preprocessed[0]
        return preprocessed

    def compile(self, strategy=None):
        self.model = Model(
            MODEL_PATH,
            self._processing.pair,
            self.target_preprocessed.y_train,
        )
        # Now we create our architecture
        branches = []
        for branch in self._branches:
            if isinstance(self.preprocessed[branch].x_train, np.ndarray):
                branches.append(
                    Branch(
                        self.preprocessed[branch].x_train,
                        self._branches[branch].nodes["NODE_TRANSFORMER"],
                        self._branches[branch].nodes["NODE_LSTM"],
                        self._branches[branch].nodes["NODE_DENSE"],
                        self._branches[branch].nodes["ATTENTION_HEADS"],
                        self._branches[branch].nodes["DROPOUT"],
                    )
                )
        main_branch = Branch(
            self.target_preprocessed.x_train,
            self._main_branch.nodes["NODE_TRANSFORMER"],
            self._main_branch.nodes["NODE_LSTM"],
            self._main_branch.nodes["NODE_DENSE"],
            self._main_branch.nodes["ATTENTION_HEADS"],
            self._main_branch.nodes["DROPOUT"],
        )
        output = Output(self._output.nodes["NODE_DENSE"], self._output.nodes["DROPOUT"])
        architecture = Architecture(branches, main_branch, output)
        self.model.compile(
            architecture,
            learning_rate=self._processing.learning_rate,
            strategy=strategy,
        )
        return self.model

    def fit(self, strategy):
        """Fit the model."""
        self.model.fit(
            strategy=strategy,
            epochs=self._processing.epochs,
            patience=self._processing.patience,
            batch_size=self._processing.batch_size,
            validation_split=self._processing.validation_split,
            patience_lr_schedule=self._processing.patience_lr_schedule,
        )

    def predict(self):
        """Predict with the model."""
        pass
