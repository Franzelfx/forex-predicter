"""Module for the Preprocessor class."""
class Preprocessor():
    """Used to preprocess the data for RNNs.
    
    @remarks The preprocessor has the following tasks:
             1. Split the data into train and test set.
             2. Scale the data.
             3. Split the data into sequences.
    """

    def __init__