#!/bin/bash
source venv/bin/activate
export API_KEY=kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo
export START_PAIR=$1
echo "API_KEY"=$API_KEY
echo "START_PAIR="$1
# Ask, if data should be taken from file
read -p "Do you want to use data from file? (y/n): " ans
if [ "$ans" != "${ans#[Yy]}" ] ;then
    echo "Data will be taken from file"
    export USE_DATA_FROM_FILE=true
else
    echo "Data will be taken from API"
    export USE_DATA_FROM_FILE=false
fi
# Run the test
screen python test/utilizer_test.py
