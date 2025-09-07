#!/bin/bash
# Complete testing guide for Pseudo-RL implementation
# This script tests all components step by step

set -e  # Exit on error

echo "======================================================================="
echo "PSEUDO-RL COMPLETE TESTING GUIDE"
echo "======================================================================="
echo ""
echo "This script will test the entire pseudo-RL pipeline step by step."
echo "Each test builds on the previous one to ensure everything works."
echo ""

# Configuration
DATASET="./intermediate_data/prepared_dataset.pth"
TEST_CHECKPOINT="./report/2025_start/20250830_232858_693d0ab_main_lr2e-04_bl1e-05_ssu0_bs4_h8_es1024_nl11_we10_as4_ad1_scosine_oadam_ge1_nkh1/Transformer_best_76.pt"

# Step 1: Verify prerequisites
echo "======================================================================="
echo "STEP 1: VERIFYING PREREQUISITES"
echo "======================================================================="
echo ""

if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset not found: $DATASET"
    echo "   Please run data preparation first"
    exit 1
fi
echo "✓ Dataset found: $DATASET"

if [ ! -f "$TEST_CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $TEST_CHECKPOINT"
    echo "   Please ensure you have a trained model checkpoint"
    exit 1
fi
echo "✓ Checkpoint found: $TEST_CHECKPOINT"

# Step 2: Test UnifiedTrainingScheduler
echo ""
echo "======================================================================="
echo "STEP 2: TESTING UNIFIED TRAINING SCHEDULER"
echo "======================================================================="
echo ""

echo "Running scheduler unit tests..."
python -m pytest tests/test_schedulars.py -v
if [ $? -eq 0 ]; then
    echo "✓ Scheduler tests passed"
else
    echo "❌ Scheduler tests failed"
    exit 1
fi

# Step 3: Test data loading with attempt sections
echo ""
echo "======================================================================="
echo "STEP 3: TESTING DATA LOADING WITH ATTEMPT SECTIONS"
echo "======================================================================="
echo ""

echo "Testing GridDataset with attempt sections..."
python -c "
from src.load_data import GridDataset
import torch

# Load dataset
dataset = torch.load('$DATASET')
grid_dataset = GridDataset()
grid_dataset.data = dataset['data'][:5]  # Test with 5 samples

# Test different attempt lengths
for attempt_len in [0, 10, 100]:
    grid_dataset.set_attempt_length(attempt_len)
    sample = grid_dataset[0]
    print(f'✓ Attempt length {attempt_len}: Task has {len(sample[\"task\"])} tokens')
"
if [ $? -eq 0 ]; then
    echo "✓ Data loading with attempt sections works"
else
    echo "❌ Data loading failed"
    exit 1
fi

# Step 4: Test trajectory generation
echo ""
echo "======================================================================="
echo "STEP 4: TESTING TRAJECTORY GENERATION"
echo "======================================================================="
echo ""

echo "Creating test trajectory directory..."
TEST_TRAJ_DIR="./test_trajectories_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_TRAJ_DIR"

echo "Generating 2 test trajectories..."
python -m src.generate_rollout \
    --checkpoint "$TEST_CHECKPOINT" \
    --dataset "$DATASET" \
    --output-dir "$TEST_TRAJ_DIR" \
    --num-samples 2 \
    --temperature 0.7 \
    --max-tokens 900

if [ $? -eq 0 ]; then
    TRAJ_COUNT=$(ls "$TEST_TRAJ_DIR"/*.pt 2>/dev/null | wc -l)
    if [ "$TRAJ_COUNT" -gt 0 ]; then
        echo "✓ Generated $TRAJ_COUNT trajectory file(s)"
    else
        echo "❌ No trajectories generated"
        exit 1
    fi
else
    echo "❌ Trajectory generation failed"
    exit 1
fi

# Step 5: Test trajectory loading
echo ""
echo "======================================================================="
echo "STEP 5: TESTING TRAJECTORY LOADING"
echo "======================================================================="
echo ""

python -c "
from src.trajectory_loader import TrajectoryLoader
loader = TrajectoryLoader('$TEST_TRAJ_DIR')
trajectories = loader.load_all_trajectories()
print(f'✓ Loaded {len(trajectories)} trajectories')
for idx, traj in trajectories.items():
    if 'attempt_tokens' in traj:
        print(f'  Sample {idx}: {len(traj[\"attempt_tokens\"])} attempt tokens')
"
if [ $? -eq 0 ]; then
    echo "✓ Trajectory loading works"
else
    echo "❌ Trajectory loading failed"
    exit 1
fi

# Step 6: Test mixed dataset
echo ""
echo "======================================================================="
echo "STEP 6: TESTING MIXED DATASET"
echo "======================================================================="
echo ""

python -c "
import torch
from src.load_data import GridDataset, MixedGridDataset
from src.trajectory_loader import TrajectoryLoader

# Load original dataset
dataset = torch.load('$DATASET')
grid_dataset = GridDataset()
grid_dataset.data = dataset['data'][:10]  # Test with 10 samples

# Load trajectories
loader = TrajectoryLoader('$TEST_TRAJ_DIR')
trajectories = loader.load_all_trajectories()

# Create mixed dataset
mixed = MixedGridDataset(grid_dataset, trajectories)
mixed.set_attempt_length(100)

# Test sampling
sample = mixed[0]
print(f'✓ Mixed dataset created with {len(mixed)} samples')
print(f'  Sample 0 has {len(sample[\"task\"])} tokens')
"
if [ $? -eq 0 ]; then
    echo "✓ Mixed dataset works"
else
    echo "❌ Mixed dataset failed"
    exit 1
fi

# Step 7: Test short training run with scheduler
echo ""
echo "======================================================================="
echo "STEP 7: TESTING SHORT TRAINING RUN WITH SCHEDULER"
echo "======================================================================="
echo ""

echo "Running 2-epoch training test..."
TEST_RUN_DIR="./runs/test_pseudo_rl_$(date +%Y%m%d_%H%M%S)"

python -m src.train \
    --dataset-files "$DATASET" \
    --load-checkpoint "$TEST_CHECKPOINT" \
    --runs-name "$(basename $TEST_RUN_DIR)" \
    --epochs 2 \
    --batch-size 2 \
    --embed-size 1024 \
    --num-layers 11 \
    --heads 8 \
    --learning-rate 1e-4 \
    --warmup-epochs 1 \
    --minimize-checkpoints

if [ $? -eq 0 ]; then
    echo "✓ Training with scheduler works"
    
    # Check if scheduler state was saved
    if [ -f "$TEST_RUN_DIR/scheduler_state.json" ]; then
        echo "✓ Scheduler state saved"
    fi
else
    echo "❌ Training failed"
    exit 1
fi

# Step 8: Test training with trajectories
echo ""
echo "======================================================================="
echo "STEP 8: TESTING TRAINING WITH TRAJECTORIES"
echo "======================================================================="
echo ""

echo "Running 1-epoch mixed training test..."
python -m src.train \
    --dataset-files "$DATASET" \
    --trajectory-folder "$TEST_TRAJ_DIR" \
    --load-checkpoint "$TEST_CHECKPOINT" \
    --runs-name "test_mixed_$(date +%Y%m%d_%H%M%S)" \
    --epochs 1 \
    --batch-size 2 \
    --embed-size 1024 \
    --num-layers 11 \
    --heads 8 \
    --learning-rate 1e-4 \
    --minimize-checkpoints

if [ $? -eq 0 ]; then
    echo "✓ Mixed training with trajectories works"
else
    echo "❌ Mixed training failed"
    exit 1
fi

# Step 9: Test orchestrator help
echo ""
echo "======================================================================="
echo "STEP 9: TESTING ORCHESTRATOR"
echo "======================================================================="
echo ""

echo "Testing orchestrator with --help..."
python -m src.orchestrate_training --help

if [ $? -eq 0 ]; then
    echo "✓ Orchestrator help works"
else
    echo "❌ Orchestrator help failed"
    exit 1
fi

# Cleanup
echo ""
echo "======================================================================="
echo "STEP 10: CLEANUP"
echo "======================================================================="
echo ""

echo "Cleaning up test files..."
rm -rf "$TEST_TRAJ_DIR"
echo "✓ Cleaned up test trajectories"

echo ""
echo "======================================================================="
echo "✅ ALL TESTS PASSED!"
echo "======================================================================="
echo ""
echo "The pseudo-RL implementation is complete and working correctly."
echo ""
echo "To run the full training pipeline:"
echo "  python -m src.orchestrate_training --dataset ./intermediate_data/prepared_dataset.pth"
echo ""
echo "To resume from a previous run:"
echo "  python -m src.orchestrate_training --resume runs/pseudo_rl_*/orchestration_state.json"
echo ""
echo "======================================================================="