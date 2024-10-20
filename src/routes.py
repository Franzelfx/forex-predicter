import math
from fastapi import APIRouter
from os import listdir
from os.path import isfile, join
import json
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from fastapi import Body
from typing import Optional
from fastapi import BackgroundTasks
from src.inference import main

router = APIRouter()

def sanitize_data(data, key_to_filter=None, value_to_filter=None):
    if isinstance(data, dict):
        for key, value in list(data.items()):  # Use list to copy keys for safe deletion
            if key == key_to_filter and value == value_to_filter:
                del data[key]  # Remove the entry if key matches and value is None
            else:
                data[key] = sanitize_data(value, key_to_filter, value_to_filter)
    elif isinstance(data, list):
        data = [sanitize_data(item, key_to_filter, value_to_filter) for item in data]
        data = [item for item in data if item]  # Remove None values after sanitization
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None  # Replace non-compliant floats with None
    return data

def filter_predictions(predictions):
    filtered_predictions = []
    previous_confidence = None

    for prediction in predictions:
        current_confidence = prediction["confidence"]
        if current_confidence != previous_confidence:
            filtered_predictions.append({
                "t": prediction["t"],
                "y_hat": prediction["y_hat"],
                "value": prediction["confidence"]
            })
        previous_confidence = current_confidence

    return filtered_predictions

def format_confidence(confidence_str):
    # Extract the numeric part of the string
    numeric_part = float(confidence_str.strip(" %"))
    # Round to one decimal place
    rounded_confidence = round(numeric_part, 1)
    # Convert back to string with percentage sign if necessary
    formatted_confidence = f"{rounded_confidence} %"
    return formatted_confidence

def get_timestamp_from_timedelta(days_delta):
    today = datetime.now(timezone.utc)
    adjusted_date = today - timedelta(days=days_delta)  # Adjusting the date by the given timedelta in days
    start_of_adjusted_date = adjusted_date.replace(hour=0, minute=0, second=0, microsecond=0)  # Start of the adjusted day
    return start_of_adjusted_date.timestamp()

@router.post("/inference")
async def manual_run(background_tasks: BackgroundTasks, pair: Optional[str] = None, gpu: Optional[int] = None):
    """
    Manually triggers the main task for predictions. 
    Allows specifying a pair and GPU.
    If no pair specified it will run for all pairs.
    If no GPU specified it will run on GPU 0.
    """
    try:
        # Schedule the main task to run in the background, passing the optional pair and gpu parameters.
        background_tasks.add_task(main, pair_name=pair, gpu=gpu)
        return {"message": "Manual run initiated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating manual run: {str(e)}")

@router.get("/recipes")
async def get_recipe_names():
    folder_path = "src/recipes"
    files = [
        f
        for f in listdir(folder_path)
        if isfile(join(folder_path, f)) and f.endswith(".json")
    ]
    # Remove the .json extension and _recipe suffix
    files = [f[:-5] for f in files]
    files = [f[:-7] if f.endswith("_recipe") else f for f in files]
    # Exclude recipe with "BASE" in the name
    files = [f for f in files if "BASE" not in f]
    return files

@router.get("/recipe/{pair}")
async def get_recipe(pair: str):
    file_path = f"src/recipes/{pair}_recipe.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as file:
        data = json.load(file)
    try:
        return data
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{e.args[0]} not found in file")


@router.get("/dumps")
async def read_dumps():
    folder_path = "src/model_predictions/composer"
    files = [
        f
        for f in listdir(folder_path)
        if isfile(join(folder_path, f)) and f.endswith(".json")
    ]
    # Remove the .json extension
    files = [f[:-5] for f in files]
    files = [f[:-5] if f.endswith("_dump") else f for f in files]
    # Sort files by name
    files.sort()
    return files


@router.get("/confidences/{pair}")
async def read_all_confidences(pair: str):
    file_path = f"src/model_predictions/composer/{pair}_dump.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(file_path, "r") as file:
        data = json.load(file)
    try:
        # Timedelta is "interval" devided by 5
        interval = data["processing"]["interval"]
        last_monday_timestamp = get_timestamp_from_timedelta(interval / 5)
        # Extract all confidence scores from predictions
        confidences = filter_predictions(data["predictions"])
        # Filter out confidences older than last Monday
        confidences = [confidence for confidence in confidences if datetime.strptime(confidence["t"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() >= last_monday_timestamp]
        # Convert timestamp strings to UTC timestamps
        for confidence in confidences:
            confidence["t"] = datetime.strptime(confidence["t"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        # Sanitize the data and remove entries where confidence is None
        sanitized_confidences = sanitize_data(confidences, key_to_filter='confidence', value_to_filter=None)
        # Filter out any remaining None entries (if necessary)
        sanitized_confidences = [conf for conf in sanitized_confidences if conf and conf.get('value') is not None]
        # Format the confidence values
        for confidence in sanitized_confidences:
            confidence["value"] = format_confidence(confidence["value"])
        return sanitized_confidences
    except KeyError:
        raise HTTPException(
            status_code=404, detail="Confidence information not found in file"
        )

@router.get("/confidence/{pair}")
async def read_latest_confidence(pair: str):
    file_path = f"src/model_predictions/composer/{pair}_dump.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as file:
        data = json.load(file)
    try:
        # Assuming each prediction in "predictions" includes a "confidence" key
        # Get the most recent confidence score
        latest_confidence = data["predictions"][-1]["confidence"]
        return latest_confidence
    except (KeyError, IndexError) as e:
        raise HTTPException(
            status_code=404, detail=f"Confidence information not found in file"
        )

@router.get("/bars/{pair}/{bars}")
async def read_model_predictions(pair: str, bars: int = 100):
    file_path = f"src/model_predictions/composer/{pair}_dump.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as file:
        data = json.load(file)

    try:
        pairs_data = data["pairs"][pair]["values"]

        # Checking if we have enough data points
        if len(pairs_data) < bars:
            raise HTTPException(status_code=400, detail="Not enough data available")

        # Get timestamp based on interval
        interval = data["processing"]["interval"]
        last_monday_timestamp = get_timestamp_from_timedelta(interval / 5)

        # Filter based on timestamp
        selected_data = [
            entry for entry in pairs_data[-bars:] 
            if datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() >= last_monday_timestamp
        ]
        # Initialize formatted_data
        formatted_data = []

        # Append existing data to formatted_data
        for entry in selected_data:
            timestamp_str = entry["time"]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
            formatted_data.append(
                {
                    "time": timestamp_utc.timestamp(),  # Convert to UTC timestamp
                    "open": entry["open"],
                    "high": entry["high"],
                    "low": entry["low"],
                    "close": entry["close"],
                }
            )
        return formatted_data
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{e.args[0]} not found in file")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/prediction/{pair}")
async def read_prediction_close(pair: str):
    file_path = f"src/model_predictions/composer/{pair}_dump.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(file_path, "r") as file:
        data = json.load(file)
    try:
        predictions = data["predictions"]
        # Get timestamp based on interval
        interval = data["processing"]["interval"]
        last_monday_timestamp = get_timestamp_from_timedelta(interval / 5)
        response_data = [
            {
                "time": datetime.strptime(prediction["t"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp(),
                "close": prediction["y_hat"],
            }
            for prediction in predictions if datetime.strptime(prediction["t"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() >= last_monday_timestamp
        ]
        return response_data
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{e.args[0]} not found in file")

@router.put("/update_prediction_times")
async def update_prediction_times(new_times: list = Body(...)):
    # Validate the input
    if not all(isinstance(time, str) for time in new_times):
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Validate the time format (hh:mm:ss)
    for time in new_times:
        try:
            datetime.strptime(time, "%H:%M:%S")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid time format. Please use hh:mm:ss")

    # Path to your app_config.json file
    config_file_path = "app_config.json"

    try:
        # Verify and read the current configuration
        with open(config_file_path, "r") as file:
            try:
                config = json.load(file)
            except json.JSONDecodeError:
                # If the JSON file is invalid, don't proceed with the update
                raise HTTPException(status_code=500, detail="Current configuration file is invalid. Update aborted.")

        # Update the DEFAULT_PREDICTION_DATETIMES field
        config["DEFAULT_PREDICTION_DATETIMES"] = new_times

        # Write the updated configuration back to the file
        with open(config_file_path, "w") as file:
            json.dump(config, file, indent=4)

        return {"message": "Prediction times updated successfully"}

    except Exception as e:
        # Handle any exceptions (e.g., file not found, JSON errors)
        raise HTTPException(status_code=500, detail=str(e))