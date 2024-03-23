import math
from fastapi import APIRouter
from os import listdir
from os.path import isfile, join
import json
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from fastapi import Body
from typing import Optional

router = APIRouter()

def sanitize_data(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = sanitize_data(value)
    elif isinstance(data, list):
        data = [sanitize_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None  # Or a specific value your application can handle
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
                "confidence": prediction["confidence"]
            })
        previous_confidence = current_confidence

    return filtered_predictions

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
        # Extract all confidence scores from predictions
        confidences = filter_predictions(data["predictions"])
        # Sanitize the data before returning it
        sanitized_confidences = sanitize_data(confidences)
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

        # Select last 'bars' data points
        selected_data = pairs_data[-bars:]

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

        # Process each prediction
        response_data = [
            {
                "time": datetime.strptime(prediction["t"], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc).timestamp(),
                "close": prediction["y_hat"],
            }
            for prediction in predictions
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

@router.post("/inferenz")
async def inferenz(pair: Optional[str] = None, gpu: Optional[int] = None):
    from src.main import main
    try:
        # Assuming 'main' is the function from your provided script
        # This will execute the main task for the specified pair and GPU (if provided)
        main(pair_name=pair, gpu=gpu)
        return {"message": f"Inferenz done for pair: {pair}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
