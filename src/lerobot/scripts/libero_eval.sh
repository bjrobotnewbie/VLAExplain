#!/bin/bash

# Set policy path and other configuration variables
POLICY_PATH=
OUTPUT_DIR=$(pwd)/src/lerobot/analyzer/analyzer_data//outputs/
DATA_DIR=$(pwd)/src/lerobot/analyzer/analyzer_data/
TASK_IDS="[0]"
N_ACTION_STEPS=10

# Export environment variable for LEROBOT_DATA_DIR
export LEROBOT_DATA_DIR="$DATA_DIR"
export TORCH_CUDAGRAPHS_DISABLE="1"


# Directly run the python command
echo "Running evaluation..."
python projects/lerobot/src/lerobot/scripts/lerobot_eval.py \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids="$TASK_IDS" \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --output_dir="$OUTPUT_DIR" \
    --env.max_parallel_tasks=1

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    exit 0
else
    echo "Error running evaluation with exit code: $?"
    exit 1
fi