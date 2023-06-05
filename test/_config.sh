#!/bin/bash
source ../venv/bin/activate
# ------------------------------
# Ask User, if multiple GPUs should be used
read -p "Do you want to use multiple GPUs? (y/n): " answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Multiple GPUs will be used"
    export USE_MULTIPLE_GPUS="True"
else
    echo "Only one GPU will be used"
    export USE_MULTIPLE_GPUS="False"
fi

read -p "Want to use pair data from saved file? (y/n): " answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Loading from saved file"
    export LOAD_FROM_SAVED_FILE=true
else
    echo "Not loading from saved file"
    export LOAD_FROM_SAVED_FILE=false
fi