#!/bin/bash
source venv/bin/activate
# Set API_KEY and START_PAIR
export API_KEY=kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo
export START_PAIR=$1
echo "API_KEY"=$API_KEY
echo "START_PAIR="$1
# Configure the env variables
./_config.sh
# Run the test
screen python test/system_test.py
