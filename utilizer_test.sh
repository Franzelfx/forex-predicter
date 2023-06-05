#!/bin/bash
source venv/bin/activate
export API_KEY=kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo
export START_PAIR=$1
echo "API_KEY"=$API_KEY
echo "START_PAIR="$1
# Ask, if data should be taken from file
input "Do you want to take data from file? (y/n)" ans
if [ $ans == "y" ]
then
    export FROM_FILE=true
else
    export FROM_FILE=false
fi
# Run the test
screen python test/utilizer_test.py
