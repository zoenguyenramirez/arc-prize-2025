#!/bin/bash
source "$(dirname "$0")/utils/suppress_warnings.sh"

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 runs/20250815_193502/Transformer_latest.pt"
    exit 1
fi

checkpoint_path="$1"

# Check if checkpoint file exists
if [ ! -f "$checkpoint_path" ]; then
    echo "Error: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

# Extract checkpoint directory and filename without extension
checkpoint_dir=$(dirname "$checkpoint_path")
checkpoint_basename=$(basename "$checkpoint_path" .pt)

# Run evaluations and save results
echo "Running ARC evaluation on: $checkpoint_path"

# Save results to file in same directory as checkpoint, with matching name
results_file="${checkpoint_dir}/${checkpoint_basename}_evaluation.txt"

# Run evaluation, display output in terminal AND save to file
python -m src.sample --checkpoint-path "$checkpoint_path" --data-source arc-agi_evaluation --second-only 2>&1 | tee "$results_file"

echo ""
echo "Results saved to: $results_file"
