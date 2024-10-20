import json

def get_config():
    with open("src/app_config.json", "r") as file:
        config = json.load(file)
    return config