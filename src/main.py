import os
import json
from pytz import utc
from fastapi import FastAPI
from src.routes import router
from src.composer import Composer
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, replace with your domain(s)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
    allow_credentials=True,  # Allow sending cookies and credentials
    expose_headers=["*"],  # Expose all response headers
)
app.include_router(router, prefix="/v1", tags=["routes"])

def get_config():
    with open("src/app_config.json", "r") as file:
        config = json.load(file)
    return config

def pairs_to_predict(config: dict):
    # Check if FOREX_PAIRS is *, then use all Pairs in the recipes folder
    # and set the prefix to "FOREEX_PREFIX", else get the pairs from the
    # FOREX_PAIRS list and set the prefix to "FOREX_PREFIX"
    if config["FOREX_PAIRS"][0] == "*":
        recipe_files = os.listdir('recipes')
        pairs = [recipe_file.replace('_recipe.json', '') for recipe_file in recipe_files if recipe_file.endswith('_recipe.json')]
        prefix = config["FOREX_PREFIX"]
    else:
        pairs = config["FOREX_PAIRS"]
        prefix = config["FOREX_PREFIX"]
    
    # Add the prefix to the pairs
    pairs = [prefix + pair for pair in pairs]
    # Remove "BASE" from the pairs
    pairs = [pair for pair in pairs if "BASE" not in pair]
    return pairs

# Main task for prediction of pairs
def main(pair_name: str = None, gpu: int = None):
    # Configure basic settings
    config = get_config()
    gpu = config["DEFAULT_GPU"] if gpu is None else gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f'Using GPU {gpu}')
    from_file = config["AQUIRE_FROM_FILE"]
    if pair_name is None:
        pairs = pairs_to_predict(config)
    else:
        pairs = [pair_name]
    # Iterate over all pairs
    for pair in pairs:
        try:
            composer_inferenz(pair, from_file=from_file)
        except Exception as e:
            print(f'Error while processing {pair}: {e}')
            continue

# Function to run the composer class for inferenz
def composer_inferenz(pair_name: str, from_file=True):
    # The composer class routine for inferenz
    composer = Composer(pair_name)
    composer.summary()
    composer.aquire(from_file=from_file)
    composer.calculate()
    composer.preprocess()
    composer.compile()
    composer.predict()
    composer.dump()

# Function to run the main task
def run_main_task():
    print("Executing main task...")
    main()  # Call your main function here

# Get the config to find out when to run the task
config = get_config()

# Create a scheduler instance
scheduler = BackgroundScheduler(timezone=utc)

# Schedule the main task based on the configuration
for time in config["DEFAULT_PREDICTION_DATETIMES"]:
    scheduler.add_job(run_main_task, 'cron', hour=int(time[:2]), minute=int(time[3:5]), second=int(time[6:]), day_of_week='mon-fri')

# Start the scheduler
scheduler.start()

if __name__ == '__main__':
    print("Pairs to predict: ", pairs_to_predict(get_config()))