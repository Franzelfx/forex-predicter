from fastapi import APIRouter
from os import listdir
from os.path import isfile, join
import json
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone


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
        return confidence
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{e.args[0]} not found in file")

@router.get("/model-predictions/{pair}/{bars}")
async def read_model_predictions(pair: str, bars: int = 100):
    file_path = f"model_predictions/json/{pair}_dump.json"
    if not isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as file:
        data = json.load(file)

    try:
        pairs_data = data["pairs"][pair]["values"]
        predictions_data = data["predictions"]["y_hat"]

        selected_columns = ["t", "o", "h", "l", "c"]
        filtered_data = {
            key: pairs_data[key][-bars:]
            for key in selected_columns
            if key in pairs_data
        }

        # Initialize formatted_data
        formatted_data = []

        # Append existing data to formatted_data
        for i in range(len(filtered_data["t"])):
            timestamp_str = filtered_data["t"][i]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
            formatted_data.append(
                {
                    "time": timestamp_utc.timestamp(),  # Convert to UTC timestamp
                    "open": filtered_data["o"][i],
                    "high": filtered_data["h"][i],
                    "low": filtered_data["l"][i],
                    "close": filtered_data["c"][i],
                }
            )

        # Determine the granularity of timestamps
        timestamps = filtered_data["t"]
        if len(timestamps) > 1:
            granularity = datetime.strptime(
                timestamps[-1], "%Y-%m-%d %H:%M:%S"
            ) - datetime.strptime(timestamps[-2], "%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("Not enough data to determine timestamp granularity")

        # Generate new timestamps for y_hat data
        last_timestamp = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
        new_timestamps = [
            int(
                (last_timestamp + (granularity * (i + 1)))
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
            for i in range(len(predictions_data))
        ]

        # Calculate dynamic open, high, and low values based on max and min of y_hat values
        y_hat_values = predictions_data
        max_y_hat = max(y_hat_values)
        min_y_hat = min(y_hat_values)
        value_range = max_y_hat - min_y_hat

        for i in range(len(predictions_data)):
            y_hat_value = predictions_data[i]
            # Calculate open, high, and low based on dynamic values
            open_value = y_hat_value - (value_range / 20)
            high_value = y_hat_value + (value_range / 20)
            low_value = y_hat_value - (value_range / 20)
        
        # Append y_hat data
        for i in range(len(predictions_data)):
            y_hat_value = predictions_data[i]
            formatted_data.append(
                {
                    "time": new_timestamps[i],
                    "open": open_value,
                    "high": high_value,
                    "low": low_value,
                    "close": y_hat_value,
                }
            )

        return formatted_data
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"{e.args[0]} not found in file")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
