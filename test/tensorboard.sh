#!/bin/bash
# Activate the virtual environment
source ../venv/bin/activate
# ----------------------------
# Directory for TensorFlow logs
LOGDIR="/home/fabian/forex-predicter/src/tensorboard"

# Running TensorBoard
tensorboard --logdir ${LOGDIR} --bind_all
