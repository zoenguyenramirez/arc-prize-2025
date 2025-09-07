#!/bin/bash

# Monitor a running training session
# Shows progress and resource usage

if [ -z "$1" ]; then
    # Find the most recent run
    RUN_DIR=$(ls -dt runs/pseudo_rl_* 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "No active runs found"
        exit 1
    fi
else
    RUN_DIR="$1"
fi

echo "Monitoring: $RUN_DIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "======================================================================="
    echo "TRAINING MONITOR - $(date)"
    echo "======================================================================="
    echo ""
    
    # Show current iteration from orchestration states
    if [ -d "$RUN_DIR/_orchestration_states" ]; then
        LATEST_STATE=$(ls -t "$RUN_DIR/_orchestration_states"/*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_STATE" ]; then
            echo "Latest state: $(basename $LATEST_STATE)"
            ITERATION=$(grep -o '"current_iteration": [0-9]*' "$LATEST_STATE" | cut -d: -f2 | xargs)
            PHASE=$(grep -o '"current_phase": "[^"]*"' "$LATEST_STATE" | cut -d'"' -f4)
            echo "Current iteration: $((ITERATION + 1))"
            echo "Current phase: $PHASE"
        fi
    fi
    
    echo ""
    echo "Directory structure:"
    ls -d "$RUN_DIR"/iter* 2>/dev/null | tail -5
    
    echo ""
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    
    echo ""
    echo "Latest checkpoint:"
    find "$RUN_DIR" -name "Transformer_latest.pt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
    
    echo ""
    echo "Disk usage:"
    du -sh "$RUN_DIR" 2>/dev/null
    
    sleep 10
done
