#!/bin/bash
source venv/bin/activate

# ------------------------------
# Set API_KEY and START_PAIR
export API_KEY=kvtkOoyqcuTgNrBqRGIhhLe766CLYbpo
export START_PAIR=$1
echo "API_KEY"=$API_KEY
echo "START_PAIR="$1

# ------------------------------
# Ask User, if multiple GPUs should be used
echo "Do you want to use multiple GPUs? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Multiple GPUs will be used"
    export USE_MULTIPLE_GPUS="True"
else
    echo "Only one GPU will be used"
    export USE_MULTIPLE_GPUS="False"
fi
screen python test/system_test.py
