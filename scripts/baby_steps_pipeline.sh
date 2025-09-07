#!/bin/bash
# Baby Steps Pipeline - Minimal configuration for fast debugging runs
# Uses production code with reduced samples for quick iteration

echo "=============================================="
echo "Baby Steps Pipeline - Debug Mode"
echo "=============================================="
echo ""
echo "Configuration:"
echo "- Dataset: Production dataset (prepared_dataset.pth)"
echo "- Iterations: 5"
echo "- Samples per epoch: 10 (fast execution)"
echo "- Model: Tiny (128 embed, 2 layers)"
echo "- Batch size: 1"
echo ""

# Set run name with timestamp
RUN_NAME="baby_steps_$(date +%Y%m%d_%H%M%S)"

# Launch orchestrator with minimal configuration
python -m src.orchestrate_training \
    --dataset ./intermediate_data/prepared_dataset.pth \
    --iterations 5 \
    --initial-epochs 10 \
    --mixed-epochs 10 \
    --samples-per-epoch 10 \
    --batch-size 1 \
    --embed-size 128 \
    --num-layers 2 \
    --heads 4 \
    --num-kv-heads 1 \
    --learning-rate 1e-4 \
    --warmup-epochs 1 \
    --accumulation-steps 1 \
    --trajectory-samples 5 \
    --temperature 0.8 \
    --dump-states \
    --verbose \
    --yes \
    2>&1 | tee "runs/${RUN_NAME}.log"

echo ""
echo "=============================================="
echo "Pipeline complete. Check runs/${RUN_NAME}/"
echo "=============================================="