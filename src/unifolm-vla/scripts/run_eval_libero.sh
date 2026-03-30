#!/bin/bash

# Get the directory where this script is located and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# From src/unifolm-vla/scripts, go up 3 levels to reach project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

# Set LIBERO environment variables
export LIBERO_HOME=
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

# Add required paths to PYTHONPATH
# 1. LIBERO home for 'import libero'
# 2. projects/unifolm-vla for 'from experiments' and 'from unifolm_vla'
UNIFOLM_VLA_PROJECT="${PROJECT_ROOT}/projects/unifolm-vla"
export PYTHONPATH="${LIBERO_HOME}:${UNIFOLM_VLA_PROJECT}:${PYTHONPATH}"

# Debug: Print paths for troubleshooting
echo "Running from directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "UNIFOLM_VLA_PROJECT: $UNIFOLM_VLA_PROJECT"
echo "PYTHONPATH: $PYTHONPATH"

export VLA_DATA_DIR=$(pwd)/src/unifolm-vla/analyzer_data/

# Set policy path and other configuration variables
POLICY_PATH=
VLM_PRETRAINED_PATH=
TASK_SUITE_NAME="libero_object"
UNORM_KEY="libero_object_no_noops"
TASK_ID=0
VIDEO_OUT_PATH="results/${TASK_SUITE_NAME}/${UNORM_KEY}/${TASK_ID}"

python projects/unifolm-vla/experiments/LIBERO/eval_libero.py \
    --args.pretrained_path="$POLICY_PATH" \
    --args.vlm_pretrained_path="$VLM_PRETRAINED_PATH" \
    --args.task_suite_name="${TASK_SUITE_NAME}" \
    --args.num_trials_per_task=1 \
    --args.window_size=2 \
    --args.video_out_path="$VIDEO_OUT_PATH" \
    --args.unnorm_key="${UNORM_KEY}" \
    --args.task_id=$TASK_ID

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    exit 0
else
    echo "Error running evaluation with exit code: $?"
    exit 1
fi