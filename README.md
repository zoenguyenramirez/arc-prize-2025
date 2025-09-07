# ARC PRIZE 2025 APPROACH
## Pseudo-Reinforcement Learning for Abstract Reasoning

---

## BACKGROUND

This work extends my 31-point solution from ARC-AGI 2024 by introducing a pseudo-reinforcement learning framework for abstract reasoning. The approach draws inspiration from [HRM (Hierarchical Reasoning Model)](https://arxiv.org/abs/2506.21734), which demonstrated that iterative refinement can achieve 32 points on ARC-AGI-1.

The key insight is that abstract reasoning can be framed as a self-supervised learning problem where the model improves through its own generated attempts, without requiring explicit reward signals or ground truth during trajectory generation.

---

## OVERVIEW

The Pseudo-RL framework introduces two critical tokens - ATTEMPT_START and ATTEMPT_END - that create a "scratch space" for reasoning. During training, the model generates solution attempts within this space, and these trajectories become training data for subsequent iterations. This creates a self-improvement loop: each iteration's model generates trajectories that help train the next iteration's model.

Unlike traditional RL which requires reward signals, this approach leverages the diversity of generated trajectories as an implicit learning signal. The model gradually develops better reasoning strategies through exposure to its own exploratory attempts, similar to how humans develop problem-solving intuition through practice.

**Token Sequence Structure:**
```
[START] [START_INPUT] grid1 [END_INPUT] [START_OUTPUT] grid2 [END_OUTPUT] 
[START_INPUT] grid3 [END_INPUT] [START_OUTPUT] grid4 [END_OUTPUT]
...
[START_INPUT] test_grid [END_INPUT] 
[ATTEMPT_START] <generated trajectory/reasoning> [ATTEMPT_END]
[START_OUTPUT] final_answer [END_OUTPUT] [END]
```

The **ATTEMPT_START** and **ATTEMPT_END** tokens (19 and 20) create the scratch space where the model explores solutions. During training, trajectories are generated in this space. During inference, the model can use this space to reason before producing the final answer.

---

## TECHNICAL APPROACH

Our framework implements a Pseudo-RL training pipeline that:

### 1. Trajectory Generation
The model generates solution attempts (trajectories) for ARC tasks, exploring different reasoning paths without needing correctness labels

### 2. Self-Supervised Learning
Uses generated trajectories from previous iterations to train the next iteration, mixing them with the original training data to expand the learning distribution

### 3. Cross-Dataset Training
Leverages multiple data sources:
- **ARC-2024**: Original ARC-AGI training tasks
- **ARC-2025**: New harder tasks for ARC-AGI-2
- **B-ARC**: Synthetic tasks with GPT-4 descriptions
- **Re-ARC**: Re-imagined variations of original tasks

---

## QUICK START

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Production Training (24-hour session)
```bash
./scripts/production_24h_run.sh
```

This will:
- Run 24 iterations of pseudo-RL training
- Each iteration takes ~1 hour
- Automatically manages checkpoints and trajectories
- Creates timestamped run folder in `runs/`

### 3. Monitor Training with TensorBoard
```bash
tensorboard --logdir=runs/
```

Open browser to `http://localhost:6006` and **switch to "WALL" mode on "Horizontal Axis"** to see:
- Training and validation loss curves across real time
- Learning rate schedule
- Attempt length progression

### 4. Evaluate Model
```bash
# Evaluate on second test cases (unseen during training)
python -m src.sample --checkpoint runs/[timestamp]/checkpoint_best.pt --second-only
```

### 5. Kaggle Deployment
Check my last year's repository for the complete Kaggle submission framework - it includes all necessary code for competition deployment.

---

## RUN STRUCTURE EXAMPLE

After running `production_24h_run.sh`, you'll get a structure like this (from an actual run):

```
runs/pseudo_rl_20250905_222406/
├── _config/                   # Run configuration
│   └── scheduler_config.json
│
├── trajectories/              # Generated trajectories from each iteration (THE KEY!)
│   ├── trajectories_iter0.pt  # ~70MB per iteration
│   ├── trajectories_iter1.pt
│   └── ...
│
├── iter0/                     # Initial training (no trajectories yet)
│   └── 20250905_222409_[git_hash]_[hyperparams]/
│       ├── events.out.tfevents.*      # TensorBoard logs
│       ├── Transformer_best_*.pt      # Best checkpoints by loss
│       ├── optimizer_state.pt         # Optimizer state
│       ├── lr_scheduler_state.pt      # LR scheduler state
│       └── model_parameters.txt       # Model config
│
├── iter1/                     # First RL iteration
│   ├── scheduler_state_prepared.json  # Iteration metadata
│   └── 20250905_230011_[git_hash]_[hyperparams]_c1/
│       └── [same structure as iter0]
│
├── iter2-23/                  # Subsequent iterations
│   └── [similar structure]
│
└── iter24/                    # Final iteration after 24 hours
    └── [final checkpoints]
```

**Key points:**
- **trajectories/** folder contains the generated trajectories - this is the core of Pseudo-RL!
- Each iteration generates ~70MB of trajectories used for next iteration's training
- Checkpoints are saved when validation loss improves
- TensorBoard logs allow real-time monitoring
- The naming convention encodes all hyperparameters for easy tracking

---

## TRAINING PIPELINE

The orchestrator manages a multi-iteration training pipeline:

**Iteration N:**
1. Load checkpoint from iteration N-1
2. Train on combined dataset:
   - Original ARC datasets
   - Generated trajectories from iteration N-1
3. Generate trajectories using the newly trained model
4. Save checkpoint and trajectories for iteration N+1
5. Repeat

Each iteration takes ~1 hour (100 epochs training + trajectory generation).
**24 iterations = 24 hours of training.**

---

## FINAL MESSAGE

**The system compiles and runs, but there may be bugs. This is a starting point - not a finished solution.**

Your mission:

- Tune hyperparameters for optimal performance  
- Push it to achieve competitive scores

Good luck! **Let's solve AGI together.**