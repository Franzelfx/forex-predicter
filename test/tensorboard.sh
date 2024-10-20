#!/bin/bash
# Activate the virtual environment
source ../venv/bin/activate

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$0")"

# Directory for TensorFlow logs (relative to the script directory)
LOGDIR="$SCRIPT_DIR/../src/tensorboard"

# Running TensorBoard
tensorboard --logdir "${LOGDIR}" --bind_all
