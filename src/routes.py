from fastapi import APIRouter
from os import listdir
from os.path import isfile, join
import json
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from fastapi import Body


router = APIRouter()


@router.get("/recipes")
async def read_recipes():
    folder_path = "recipes"
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


@router.get("/dumps")
async def read_dumps():
    folder_path = "model_predictions/json"
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


@router.get("/confidence/{pair}")
async def read_confidence(pair: str):
    file_path = f"model_predictions/json/{pair}_dump.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as file:
        data = json.load(file)
    try:
        confidence = data["confidence"]
        # Round to 2 decimal places, confidence is a string with % sign
        confidence = round(float(confidence[:-2]), 2)
        # Convert back to string and append % sign
        confidence = str(confidence) + " %"
        return confidence
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{e.args[0]} not found in file")


@router.get("/bars/{pair}/{bars}")
async def read_model_predictions(pair: str, bars: int = 100):
    file_path = f"model_predictions/json/{pair}_dump.json"
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
    file_path = f"model_predictions/json/{pair}_dump.json"
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