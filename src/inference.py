
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc
from src.composer import Composer
from loguru import logger as loguru
from src.config import get_config

def pairs_to_predict(config: dict):
    # Check if FOREX_PAIRS is *, then use all Pairs in the recipes folder
    # and set the prefix to "FOREEX_PREFIX", else get the pairs from the
    # FOREX_PAIRS list and set the prefix to "FOREX_PREFIX"
    if config["FOREX_PAIRS"][0] == "*":
        recipe_files = os.listdir('src/recipes')
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
async def main(pair_name: str = None, gpu: int = None):
    config = get_config()
    gpu = config["DEFAULT_GPU"] if gpu is None else gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    loguru.info(f'Using GPU {gpu}')
    from_file = config["AQUIRE_FROM_FILE"]
    if pair_name is None:
        pairs = pairs_to_predict(config)
    else:
        pairs = [pair_name]
    
    for pair in pairs:
        await composer_inferenz_async(pair, from_file=from_file)

async def run_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, func, *args)
    return result

async def composer_inferenz_async(pair_name: str, from_file=True):
    # Call the synchronous composer_inferenz in an async way
    await run_in_executor(composer_inferenz, pair_name, from_file)

# Function to run the composer class for inferenz
def composer_inferenz(pair_name: str, from_file=True):
    # The composer class routine for inferenz
    composer = Composer(pair_name)
    composer.summary()
    composer.aquire(from_file=from_file, ignore_start=True)
    composer.calculate()
    composer.preprocess()
    composer.compile()
    composer.predict()
    composer.dump()

    # Function to run the main task
async def run_main_task():
    try:
        loguru.info('Starting main task')
        await main()  # Call your main function here
    except Exception as e:
        loguru.error(f'Error while running main task: {e}')


def init_scheduler():
    # Get the config to find out when to run the task
    config = get_config()
    # Create a scheduler instance
    scheduler = BackgroundScheduler(timezone=utc)
    # Add the job to the scheduler
    for time in config["DEFAULT_PREDICTION_DATETIMES"]:
        loguru.info(f'Adding job to scheduler for time: {time}')
        scheduler.add_job(run_main_task, 'cron', hour=int(time[:2]), minute=int(time[3:5]), second=int(time[6:]), day_of_week='mon-fri')
    # Start the scheduler
    scheduler.start()
    loguru.info('Scheduler started')
    return scheduler
