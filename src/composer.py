"""This module composes data_aquirer, indicator, preprocessor and model into a single class."""

class Composer():
    """This class takes some recipe.json file and composes the data_aquirer, indicator, preprocessor and model into a single class."""
    
    def __init__(self, recipe):
        """Set the fundamental attributes.
    
        @param recipe: The recipe.json file.
        """
        self._recipe = recipe
    
    def compose(self):
        """Compose the data_aquirer, indicator, preprocessor and model into a single class."""
