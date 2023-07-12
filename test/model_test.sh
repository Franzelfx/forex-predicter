#!/bin/bash
# Set API_KEY and START_PAIR
export API_KEY=kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo
export START_PAIR=$1
echo "API_KEY"=$API_KEY
echo "START_PAIR="$1
# If no start pair is given, use default
if [ -z "$START_PAIR" ]
then
    export START_PAIR="CADJPY"
fi
# Configure the env variables
source ../venv/bin/activate
# ------------------------------
# Ask User, if multiple GPUs should be used
read -p "Do you want to use multiple GPUs? y/[n]: " answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Multiple GPUs will be used"
    export USE_MULTIPLE_GPUS="True"
else
    echo "Only one GPU will be used"
    export USE_MULTIPLE_GPUS="False"
fi

read -p "Want to use pair data from saved file? y/[n]: " answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Loading from saved file"
    export FROM_SAVED_FILE=true
else
    echo "Not loading from saved file"
fi

read -p "Want to utilize the saved model? y/[n]: " answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Utilizing the saved model"
    export UTILIZE_MODEL="True"
else
    echo "Not utilizing the saved model"
    export UTILIZE_MODEL="False"
fi
export TENSORBOARD_LOGDIR="./tensorboard"
screen -S model_test bash -c 'python -u model_test.py'