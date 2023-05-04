"""This module composes data_aquirer, indicator, preprocessor and model into a single class."""
import os
import json
import pandas as pd
from tabulate import tabulate
from dataclasses import dataclass
from .indicators import Indicators
from .data_aquirer import Data_Aquirer
from .preprocessor import Preprocessor
from .branched_model import Branched_Model

PATH = os.path.abspath(os.path.dirname(__file__))
PATH_PAIRS = os.path.abspath(os.path.join(PATH, "pairs"))
PATH_INDICATORS = os.path.abspath(os.path.join(PATH, "indicators"))
PATH_RECIPES = os.path.abspath(os.path.join(PATH, "recipes"))
PATH_MODEL = os.path.abspath(os.path.dirname(__file__))

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
        self.target_feature = processing_sec["TARGET_FEATURE"]

@dataclass
class Branch:
    """This class is used to store the branching attributes."""

    def __init__(self, name: str, attributes: dict):
        """Set the attributes."""
        self.name = name
        self.indicators = attributes["INDICATORS"]
        self.conv_nodes = attributes["CONV_NODES"]
        self.lstm_nodes = attributes["LSTM_NODES"]
        self.dense_nodes = attributes["DENSE_NODES"]
        self.dropout = attributes["DROPOUT"]


@dataclass
class Concat:
    """This class is used to store the concatenation attributes."""

    def __init__(self, json: dict):
        """Set the attributes."""
        # Top level
        concat_sec = json["CONCAT"]
        # Concat
        self.conv_nodes = concat_sec["CONV_NODES"]
        self.lstm_nodes = concat_sec["LSTM_NODES"]
        self.dense_nodes = concat_sec["DENSE_NODES"]
        self.dropout = concat_sec["DROPOUT"]


class Composer:
    """This class takes some recipe.json file and composes the data_aquirer, indicator, preprocessor and model into a single class."""

    def __init__(self, pair_name: str):
        """Set the fundamental attributes.

        @param recipe: Path to the recipe.json file.
        """
        # Set the fundamental attributes to default values
        self._processing = None
        self._branches = {}
        self._concat = None
        self._training = None

        recipe_path = os.path.abspath(os.path.join(PATH_RECIPES ,f"{pair_name}_recipe.json"))
        base_recipe_path = os.path.abspath(os.path.join(PATH_RECIPES, "__BASE__recipe.json"))
    
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
        self._concat = Concat(recipe)

    def _model_branches(self) -> str:
        """Return the mode tree of the attributes."""
        header = [""]
        table = {}
        # remove the name attribute
        for branch in self._branches:
            table[""] = self._branches[branch].__dict__
            # pop the name and indicators attribute
            table[""].pop("name")
            table[""].pop("indicators")
            break

        for branch in self._branches:
            header.append(branch)
            attr_list = []
            for key, value in self._branches[branch].__dict__.items():
                if key != "name" and key != "indicators":
                    attr_list.append(value)
            table[branch] = attr_list
        return tabulate(table, headers=header, tablefmt="rst")

    def _model_concat(self) -> str:
        """Return the mode tree of the attributes."""
        header = ["", "CONCAT"]
        table = {}
        # remove the name attribute
        table[""] = self._concat.__dict__
        attr_list = []
        for key, value in self._concat.__dict__.items():
            if key != "name":
                attr_list.append(value)
        table['CONCAT'] = attr_list
        return tabulate(table, headers=header, tablefmt="rst")

    def summary(self):
        """Print the summary of the attributes."""
        print(self._model_branches())
        print(self._model_concat())

    def compose(self):
        """Compose the data_aquirer, indicator, preprocessor and model into a single class."""
        # data_collection = {}
        # for branch in self._branches:
        #     # Get the data for every single branch
        #     aquirer = Data_Aquirer(PATH_PAIRS, self._base.api_key, api_type=self._base.api_type)
        #     data = aquirer.get(branch, self._processing.interval, save=True)
        #     # Get the indicators
        #     indicators = Indicators(data, self._branches[branch].indicators)
        #     data = indicators.calculate(save=True, path=f"{PATH_INDICATORS}/{branch}_{self._processing.interval}.csv")
        #     # Get the preprocessor
        #     preprocessor = Preprocessor(data, self._processing.steps_in, self._processing.steps_out, self._processing.test_length)
        #     preprocessor.summary()
        #     data_collection[branch] = preprocessor
        # Create the model
        # TODO first compose with one branch (simpler)
        model = Branched_Model(PATH_MODEL, self._processing.pair, x_train=None, y_train=None)
        model.add_branch(conv=[64, 32], lstm=[32, 16], dense=[16, 8], dropout=0.2)
        model.compile_and_fit()