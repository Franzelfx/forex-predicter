#!/bin/bash
# Set API_KEY and START_PAIR
export API_KEY=kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo
export START_PAIR=$1
echo "API_KEY"=$API_KEY
echo "START_PAIR="$1
# Configure the env variables
source ../venv/bin/activate
# ------------------------------
# Ask if want to use saved file
read -p "Want to use pair data from saved file? (y/n): " answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Loading from saved file"
    export FROM_SAVED_FILE=true
else
    echo "Not loading from saved file"
fi
screen python utilizer_test.py
