#!/bin/bash

# Resume an interrupted production run
# Usage: ./production_resume.sh <run_directory>

if [ -z "$1" ]; then
    echo "Usage: $0 <run_directory>"
    echo "Example: $0 runs/pseudo_rl_20250903_123456"
    exit 1
fi

RUN_DIR="$1"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory $RUN_DIR does not exist"
    exit 1
fi

echo "======================================================================="
echo "RESUMING PRODUCTION PSEUDO-RL TRAINING"
echo "======================================================================="
echo "Resume time: $(date)"
echo "Run directory: $RUN_DIR"
echo ""

python -m src.orchestrate_training --resume "$RUN_DIR"

echo ""
echo "======================================================================="
echo "Training complete!"
echo "End time: $(date)"
echo "======================================================================="
