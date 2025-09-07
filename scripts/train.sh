#!/bin/bash

# Set PyTorch CUDA memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Source GPU memory check if available
if [ -f "$(dirname "$0")/utils/check_gpu_memory.sh" ]; then
    source "$(dirname "$0")/utils/check_gpu_memory.sh"
fi

# Function to train the model
train_model() {
    echo "Training dense transformer model with MQA - Memory optimized..."
    python -m src.train \
        --dataset-files intermediate_data/prepared_dataset_arc_2024.pth \
                       intermediate_data/prepared_dataset_arc_2025.pth \
                       intermediate_data/prepared_dataset_barc.pth \
                       intermediate_data/prepared_dataset_rearc.pth \
        --epochs 80 \
        --embed-size 1024 \
        --num-layers 11 \
        --heads 8 \
        --batch-size 4 \
        --learning-rate 2e-4 \
        --max-seq-length 2700 \
        --accumulation-steps 4 \
        --num-kv-heads 1
}

# Main execution
main() {
    train_model
}

# Run the main function
main "$@"
