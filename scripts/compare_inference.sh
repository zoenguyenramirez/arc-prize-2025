#!/bin/bash

# Compare outputs between src.sample and RL trajectory generator
# Both process the same data for validation

source "$(dirname "$0")/utils/suppress_warnings.sh"

# Configuration
CHECKPOINT_PATH=""
DATA_SOURCE="arc-agi_evaluation"

# Parse arguments
usage() {
    echo "Usage: $0 --checkpoint PATH [options]"
    echo ""
    echo "Compare src.sample vs RL trajectory generator outputs"
    echo ""
    echo "Required:"
    echo "  --checkpoint PATH         Path to model checkpoint"
    echo ""
    echo "Options:"
    echo "  --data-source SOURCE      Data source (default: arc-agi_evaluation)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --checkpoint runs/model.pt"
    echo "  $0 --checkpoint model.pt --data-source arc-agi_training"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --data-source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate checkpoint
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint is required"
    usage
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Create output directories in temp folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="temp/comparison_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "INFERENCE COMPARISON TEST"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data source: $DATA_SOURCE"
echo "Temperature: $TEMPERATURE (deterministic)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Step 1: Run src.sample (original)
echo "Step 1: Running src.sample (original implementation)..."
echo "-----------------------------------------"

SAMPLE_OUTPUT="${OUTPUT_DIR}/sample_output.pt"
SAMPLE_LOG="${OUTPUT_DIR}/sample_output.log"
python -m src.sample \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --data-source "$DATA_SOURCE" \
    --second-only \
    --output-file "$SAMPLE_OUTPUT" \
    2>&1 | tee "$SAMPLE_LOG"

echo ""
echo "src.sample output saved to: $SAMPLE_OUTPUT"
echo "Log saved to: $SAMPLE_LOG"
echo ""

# Step 2: Run RL trajectory generator
echo "Step 2: Running RL trajectory generator (with KV cache)..."
echo "-----------------------------------------"

TRAJECTORY_OUTPUT="${OUTPUT_DIR}/trajectory_output.txt"
./scripts/generate_trajectories.sh \
    --checkpoint "$CHECKPOINT_PATH" \
    --data-source "$DATA_SOURCE" \
    --second-only \
    --batch-size 1 \
    --output-dir "$OUTPUT_DIR/trajectories" \
    2>&1 | tee "$TRAJECTORY_OUTPUT"
# NOTE: Will process all 52 sequences with batch_size=1

echo ""
echo "Trajectory generator output saved to: $TRAJECTORY_OUTPUT"
echo ""

# Step 3: Run comparison analysis
echo "Step 3: Running comparison analysis..."
echo "-----------------------------------------"

# Find the trajectory file that was generated
TRAJ_FILE=$(find "$OUTPUT_DIR/trajectories" -name "trajectories_*.pt" -type f | head -1)

if [ -f "$SAMPLE_OUTPUT" ] && [ -f "$TRAJ_FILE" ]; then
    echo "Comparing outputs..."
    # Run with verbose by default and save report
    python tests/compare_inference_outputs.py "$SAMPLE_OUTPUT" "$TRAJ_FILE" --verbose --output "$OUTPUT_DIR/comparison_report.pt"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Comparison completed successfully!"
        echo "Report saved to: $OUTPUT_DIR/comparison_report.pt"
    else
        echo ""
        echo "❌ Comparison failed - check the error above"
    fi
else
    echo "Error: Output files not found"
    echo "  Sample output: $SAMPLE_OUTPUT (exists: $([ -f "$SAMPLE_OUTPUT" ] && echo yes || echo no))"
    echo "  Trajectory file: $TRAJ_FILE (exists: $([ -f "$TRAJ_FILE" ] && echo yes || echo no))"
    exit 1
fi

echo ""
echo "=========================================="
echo "RESULTS SAVED"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo "  • Sample output: sample_output.pt"
echo "  • Trajectory output: trajectories/*.pt" 
echo "  • Comparison report: comparison_report.pt"
echo "  • Logs: *.log"
echo ""