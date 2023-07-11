#!/bin/bash
# Activate the virtual environment
source ../venv/bin/activate
# ----------------------------
# Directory for TensorFlow logs
LOGDIR="./tensorflow"

# Ensure the log directory exists
mkdir -p ${LOGDIR}

# Running TensorBoard
tensorboard --logdir ${LOGDIR} --bind_all
