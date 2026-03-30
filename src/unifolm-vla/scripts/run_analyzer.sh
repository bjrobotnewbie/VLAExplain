#!/bin/bash

# Set environment variable for LEROBOT_DATA_DIR
export VLA_DATA_DIR=$(pwd)/src/unifolm-vla/analyzer_data/
export TOKENIZER_PATH=

# Configuration variables
HOST="0.0.0.0"
PORT=7862
N_ACTION_STEPS=8

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER_DIR="${SCRIPT_DIR}/../analyzer"

# Change to analyzer directory
cd "$ANALYZER_DIR" || {
    echo "Error: Failed to change directory to $ANALYZER_DIR"
    exit 1
}

# Run the analyzer script with specified arguments
python main.py \
    --host $HOST \
    --port $PORT \
    --n_action_steps $N_ACTION_STEPS

# Capture exit code
EXIT_CODE=$?

# Return to original directory
cd - > /dev/null || {
    echo "Warning: Failed to return to original directory"
}

# Exit with the same code as the Python script
exit $EXIT_CODE