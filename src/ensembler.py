import os
import json
from src.composer import Composer

PATH = os.path.dirname(__file__)

class Ensembler:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self._load_data()

    def _load_data(self):
        """Load the data from the given path."""
        try:
            with open(self.path, "r") as file:
                self.data = json.load(file)
        except Exception as e:
            print(f"Error loading ensemble data: {e}")

    def _create_composers(self):
        """Create composer instances for each model in the ensemble."""
        composers = {}
        for pair, models in self.data.items():
            composers[pair] = [Composer(model) for model in models['models']]
        return composers

    def predict(self, from_file=True):
        """Predict using all models for each forex pair."""
        composers = self._create_composers()
        for pair, composer_list in composers.items():
            for composer in composer_list:
                try:
                    composer.summary()
                    composer.aquire(from_file=from_file, ignore_start=True)
                    composer.calculate()
                    composer.preprocess()
                    composer.compile()
                    composer.predict()
                    composer.dump()
                except Exception as e:
                    print(f"Error predicting for {pair} using {composer}: {e}")

if __name__ == "__main__":
    ensemble_description_path = os.path.join(PATH, "app_ensemble_description.json")
    ensembler = Ensembler(ensemble_description_path)
    ensembler.predict()