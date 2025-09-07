#!/usr/bin/env python3
"""
Fixed Pseudo-RL Training Orchestrator
Implements correct design: One training phase per iteration using previous iteration's trajectories
"""

import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

class OrchestrationState:
    """Manages immutable orchestration state"""
    
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.state_dir = self.run_dir / '_orchestration'
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> Optional[Dict]:
        """Load the most recent state"""
        # Look for main state files (not the event-specific ones)
        state_files = []
        for f in self.state_dir.glob('*.json'):
            # Skip event-specific files (those with event names after timestamp)
            if not any(event in f.stem for event in ['_iteration_', '_training_', '_rollout_', '_complete']):
                state_files.append(f)
        
        state_files = sorted(state_files)
        if not state_files:
            return None
        
        latest_state = state_files[-1]
        with open(latest_state, 'r') as f:
            return json.load(f)
    
    def save(self, state: Dict):
        """Save state to new timestamped file (immutable)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        state_file = self.state_dir / f"{timestamp}.json"
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def is_phase_completed(self, iteration: int, phase: str) -> bool:
        """Check if a specific phase is completed"""
        state = self.load()
        if not state:
            return False
        phase_key = f'iter_{iteration}_{phase}'
        completed_phases = state.get('completed_phases', {})
        if not isinstance(completed_phases, dict):
            # Handle corrupted state
            return False
        return phase_key in completed_phases
    
    def mark_phase_completed(self, state: Dict, iteration: int, phase: str, **kwargs):
        """Mark a phase as completed with metadata"""
        if 'completed_phases' not in state:
            state['completed_phases'] = {}
        
        phase_key = f'iter_{iteration}_{phase}'
        state['completed_phases'][phase_key] = {
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.save(state)


class FixedPseudoRLOrchestrator:
    """Fixed orchestrator with correct single-training-phase design"""
    
    def __init__(self, args):
        self.args = args
        self.state = None
        self.state_manager = None
        self.run_dir = None
        self.scheduler_config_path = None
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Execute a command with proper error handling"""
        print(f"\n📍 {description}")
        print(f"   Command: {' '.join(cmd[:3])}...")
        print(f"   " + "="*60)
        
        try:
            # Stream output directly to terminal
            result = subprocess.run(cmd, check=True)
            print(f"   " + "="*60)
            print(f"   ✓ Success")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   " + "="*60)
            print(f"   ❌ Failed with code {e.returncode}")
            return False
    
    def dump_state(self, event: str, data: Dict, iteration: int = None, phase: str = None):
        """Dump state for debugging and traceability"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        dump_data = {
            'timestamp': timestamp,
            'event': event,
            'iteration': iteration,
            'phase': phase,
            **data
        }
        
        state_file = self.run_dir / '_orchestration' / f'{timestamp}_{event}.json'
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(dump_data, f, indent=2)
    
    def prepare_scheduler_state(self, prev_iteration: int, curr_iteration: int):
        """Prepare scheduler state for next iteration with trajectory marker"""
        # Find scheduler state from previous iteration's training
        prev_iter_dir = self.run_dir / f'iter{prev_iteration}'
        
        # Look for the most recent training subdirectory's scheduler state
        latest_state_file = None
        for subdir in sorted(prev_iter_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith('202'):
                candidate = subdir / 'scheduler_state.json'
                if candidate.exists():
                    latest_state_file = candidate
        
        if not latest_state_file:
            print(f"⚠️  No scheduler state found from iteration {prev_iteration}")
            return None
        
        # Read the state
        with open(latest_state_file, 'r') as f:
            scheduler_state = json.load(f)
        
        # Mark trajectory start if not already marked
        if scheduler_state.get('samples_at_trajectory_start') is None:
            scheduler_state['samples_at_trajectory_start'] = scheduler_state['total_samples']
            print(f"   Marking trajectory start at {scheduler_state['total_samples']} samples")
        
        # Set the iteration number for iteration-based attempt length calculation
        scheduler_state['current_iteration'] = curr_iteration
        
        # Write prepared state for next iteration
        next_iter_dir = self.run_dir / f'iter{curr_iteration}'
        next_iter_dir.mkdir(parents=True, exist_ok=True)
        prepared_state_file = next_iter_dir / 'scheduler_state_prepared.json'
        
        with open(prepared_state_file, 'w') as f:
            json.dump(scheduler_state, f, indent=2)
        
        print(f"   Created prepared scheduler state: {prepared_state_file}")
        return str(prepared_state_file)
    
    def run_training(self, iteration: int) -> Optional[str]:
        """Run training for one iteration (with trajectories if available)"""
        config = self.state['config']
        
        # Determine trajectory source (from PREVIOUS iteration)
        trajectory_dir = None
        scheduler_previous_state = None
        
        if iteration > 0:
            # Use shared trajectory directory that accumulates all iterations
            shared_traj_dir = self.run_dir / 'trajectories'
            if shared_traj_dir.exists():
                trajectory_dir = str(shared_traj_dir)
                print(f"   Using accumulated trajectories from: {trajectory_dir}")
            
            # Prepare scheduler state with trajectory marker
            scheduler_previous_state = self.prepare_scheduler_state(iteration - 1, iteration)
        
        # Decay learning rate for later iterations
        current_lr = config['learning_rate'] / (2 ** iteration)
        
        # Build training command
        cmd = [
            'python', '-m', 'src.train',
            '--dataset-files'] + config['dataset'] + [
            '--runs-name', f"{self.state['run_name']}/iter{iteration}",
            '--batch-size', str(config['batch_size']),
            '--embed-size', str(config['embed_size']),
            '--num-layers', str(config['num_layers']),
            '--heads', str(config['heads']),
            '--learning-rate', str(current_lr),
            '--accumulation-steps', str(config.get('accumulation_steps', 4)),
            '--num-kv-heads', str(config.get('num_kv_heads', 1)),
            '--max-seq-length', str(config.get('max_seq_length', 2700)),
            '--minimize-checkpoints',
            '--scheduler-config', str(self.scheduler_config_path)
        ]
        
        # Add trajectory folder if available
        if trajectory_dir:
            cmd.extend(['--trajectory-folder', trajectory_dir])
        
        # Add previous scheduler state if available
        if scheduler_previous_state:
            cmd.extend(['--scheduler-previous-state', scheduler_previous_state])
        
        # Determine number of epochs
        if iteration == 0:
            epochs = config['initial_epochs']
        else:
            epochs = config['mixed_epochs']
        cmd.extend(['--epochs', str(epochs)])
        
        # Add samples-per-epoch if specified
        if config.get('samples_per_epoch'):
            cmd.extend(['--samples-per-epoch', str(config['samples_per_epoch'])])
        
        # Load checkpoint if continuing from previous iteration
        if iteration > 0 and 'latest' in self.state.get('checkpoints', {}):
            cmd.extend(['--load-checkpoint', self.state['checkpoints']['latest']])
        
        # Dump training state
        self.dump_state("training_start", {
            "iteration": iteration,
            "has_trajectories": trajectory_dir is not None,
            "trajectory_dir": trajectory_dir,
            "learning_rate": current_lr,
            "epochs": epochs,
            "scheduler_state": scheduler_previous_state
        }, iteration, "training")
        
        # Run training
        phase_name = "training_with_trajectories" if trajectory_dir else "initial_training"
        success = self.run_command(cmd, f"Training iteration {iteration} ({phase_name})")
        
        if success:
            # Find the checkpoint
            iter_dir = self.run_dir / f'iter{iteration}'
            checkpoint_path = None
            
            for subdir in sorted(iter_dir.iterdir()):
                if subdir.is_dir() and subdir.name.startswith('202'):
                    candidate = subdir / 'Transformer_latest.pt'
                    if candidate.exists():
                        checkpoint_path = candidate
            
            if checkpoint_path:
                checkpoint_str = str(checkpoint_path)
                self.state['checkpoints'][f'iteration_{iteration}'] = checkpoint_str
                self.state['checkpoints']['latest'] = checkpoint_str
                self.state_manager.mark_phase_completed(self.state, iteration, 'training',
                                                       checkpoint=checkpoint_str)
                
                # Dump completion
                self.dump_state("training_complete", {
                    "checkpoint": checkpoint_str,
                    "checkpoint_size": checkpoint_path.stat().st_size
                }, iteration, "training_done")
                
                return checkpoint_str
            else:
                print(f"⚠️  Warning: Could not find checkpoint after training")
                return None
        
        return None
    
    def generate_trajectories(self, iteration: int, checkpoint: str) -> Optional[str]:
        """Generate rollout trajectories"""
        config = self.state['config']
        # Use a single shared trajectory directory for all iterations
        traj_dir = self.run_dir / 'trajectories'
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Dump rollout state
        self.dump_state("rollout_start", {
            "iteration": iteration,
            "checkpoint": checkpoint,
            "trajectory_dir": str(traj_dir),
            "num_samples": config.get('trajectory_samples', 1000)
        }, iteration, "rollout")
        
        # Create iteration-specific output filename
        output_file = traj_dir / f'trajectories_iter{iteration}.pt'
        
        cmd = [
            'python', '-m', 'src.generate_rollout',
            '--checkpoint', checkpoint,
            '--dataset'] + config['dataset'] + [  # Pass each dataset file as separate argument
            '--output-dir', str(traj_dir),
            '--output-file', str(output_file),  # Specify exact filename for this iteration
            '--num-samples', str(config.get('trajectory_samples', 1000)),
            '--temperature', str(config.get('temperature', 0.8))
        ]
        
        success = self.run_command(cmd, f"Generating trajectories for iteration {iteration}")
        
        if success:
            traj_dir_str = str(traj_dir)
            self.state['trajectories'][f'iteration_{iteration}'] = traj_dir_str
            self.state_manager.mark_phase_completed(self.state, iteration, 'trajectory_generation',
                                                   trajectory_dir=traj_dir_str)
            
            # Dump completion
            self.dump_state("rollout_complete", {
                "trajectory_dir": traj_dir_str,
                "num_files": len(list(traj_dir.glob("*.json")))
            }, iteration, "rollout_done")
            
            return traj_dir_str
        
        return None
    
    def generate_timeline_preview(self, num_iterations, samples_per_iter, attempt_lengths):
        """Generate comprehensive training timeline preview PNG"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            
            # Create figure with 4 subplots
            fig = plt.figure(figsize=(18, 14))
            
            # Create grid: 3 rows, 2 columns
            # Top row: Timeline gantt chart (spans both columns)
            # Middle left: Attempt length progression
            # Middle right: Dataset composition
            # Bottom: Learning curve placeholder (spans both columns)
            
            ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)  # Timeline
            ax2 = plt.subplot2grid((3, 2), (1, 0))  # Attempt length
            ax3 = plt.subplot2grid((3, 2), (1, 1))  # Dataset composition
            ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)  # Learning curve placeholder
            
            iterations = list(range(len(attempt_lengths)))
            
            # Plot 1: Pseudo-RL Timeline Gantt Chart
            ax1.set_title('Pseudo-RL Training Timeline', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (hours)', fontsize=12)
            ax1.set_ylabel('Iteration', fontsize=12)
            
            # Estimate time per iteration (~1 hour)
            hours_per_iter = 1.0
            
            for i in range(min(num_iterations, 24)):  # Show up to 24 iterations
                # Training phase (blue)
                train_start = i * hours_per_iter
                train_duration = hours_per_iter * 0.8  # 80% for training
                ax1.barh(i, train_duration, left=train_start, height=0.8, 
                        color='royalblue', alpha=0.7, label='Training' if i == 0 else '')
                
                # Rollout phase (green)
                rollout_start = train_start + train_duration
                rollout_duration = hours_per_iter * 0.2  # 20% for rollout
                ax1.barh(i, rollout_duration, left=rollout_start, height=0.8,
                        color='green', alpha=0.7, label='Rollout' if i == 0 else '')
                
                # Add attempt length annotation
                if i == 0:
                    label_text = 'No traj'
                else:
                    label_text = f'L={attempt_lengths[i]}'
                ax1.text(train_start + hours_per_iter/2, i, label_text, 
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            
            ax1.set_xlim(0, num_iterations * hours_per_iter)
            ax1.set_ylim(-0.5, min(num_iterations, 24) - 0.5)
            ax1.grid(True, alpha=0.3, axis='x')
            ax1.legend(loc='upper right')
            ax1.invert_yaxis()  # Show iteration 0 at top
            
            # Plot 2: Attempt Length Progression
            ax2.plot(iterations, attempt_lengths, 'b-', linewidth=2, marker='o', markersize=6)
            ax2.fill_between(iterations, 0, attempt_lengths, alpha=0.3)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Max Attempt Length', fontsize=12)
            ax2.set_title('Linear Attempt Length Growth (+60/iter)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-0.5, len(iterations) - 0.5)
            ax2.set_ylim(-50, 1100)
            ax2.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Max (1000)')
            ax2.legend()
            
            # Plot 3: Sampling Composition Per Iteration
            ax3.set_title('Training Sample Composition Per Iteration', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('Samples Used in Training', fontsize=12)
            
            # Get actual values
            original_samples = 640000  # Total original dataset size
            trajectory_samples = self.state['config'].get('trajectory_samples', 1000)
            
            # Calculate what we actually sample per iteration
            original_per_iter = []
            rollout_per_iter = []
            rollout_percentage = []
            
            for i in range(num_iterations):
                if i == 0:
                    # No trajectories in iteration 0
                    original_per_iter.append(samples_per_iter)
                    rollout_per_iter.append(0)
                    rollout_percentage.append(0)
                else:
                    # Available trajectories accumulate
                    available_trajectories = min(i * trajectory_samples, 100000)  # Cap for visualization
                    
                    # In reality, we sample randomly from the combined pool
                    # The probability of getting a trajectory sample depends on the pool size
                    total_pool = original_samples + available_trajectories
                    traj_probability = available_trajectories / total_pool
                    
                    # Expected samples of each type
                    expected_traj = samples_per_iter * traj_probability
                    expected_orig = samples_per_iter * (1 - traj_probability)
                    
                    original_per_iter.append(expected_orig)
                    rollout_per_iter.append(expected_traj)
                    rollout_percentage.append(traj_probability * 100)
            
            # Stacked bar chart
            width = 0.8
            x = np.arange(num_iterations)
            
            ax3.bar(x, original_per_iter, width, label='Original Dataset', color='blue', alpha=0.7)
            ax3.bar(x, rollout_per_iter, width, bottom=original_per_iter, 
                   label='Rollout Trajectories', color='green', alpha=0.7)
            
            # Add percentage annotations on bars
            for i in range(num_iterations):
                if i > 0:
                    y_pos = original_per_iter[i] + rollout_per_iter[i] / 2
                    if rollout_percentage[i] > 0.1:  # Only show if significant
                        ax3.text(i, y_pos, f'{rollout_percentage[i]:.1f}%', 
                                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            
            # Add horizontal line for total samples per iteration
            ax3.axhline(y=samples_per_iter, color='red', linestyle='--', alpha=0.5, 
                       label=f'Total per iter: {samples_per_iter:,}')
            
            # Add text box with key insight
            # Calculate final percentage for display
            final_trajectories = num_iterations * trajectory_samples
            final_pool = original_samples + final_trajectories
            final_percentage = (final_trajectories / final_pool) * 100
            
            insight_text = (f"With {trajectory_samples:,} trajectories/iter\n"
                          f"and {samples_per_iter:,} samples/iter,\n"
                          f"trajectories reach {final_percentage:.1f}% by iter {num_iterations}")
            
            # Choose color based on percentage
            text_color = 'green' if final_percentage > 5 else 'orange' if final_percentage > 2 else 'red'
            bg_color = 'lightgreen' if final_percentage > 5 else 'yellow' if final_percentage > 2 else 'pink'
            
            ax3.text(0.98, 0.98, insight_text, transform=ax3.transAxes,
                    fontsize=11, color=text_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.7),
                    ha='right', va='top')
            
            ax3.set_xlim(-0.5, num_iterations - 0.5)
            ax3.set_ylim(0, samples_per_iter * 1.1)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
            
            # Plot 4: Learning Curve Placeholder
            ax4.text(0.5, 0.5, 'Learning Curve\n(Will be populated during training)', 
                    ha='center', va='center', fontsize=14, color='gray', 
                    transform=ax4.transAxes)
            ax4.set_xlabel('Iteration', fontsize=12)
            ax4.set_ylabel('Loss', fontsize=12)
            ax4.set_title('Training Progress', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, num_iterations)
            
            # Overall title
            fig.suptitle(f'24-Hour Pseudo-RL Training Plan: {num_iterations} Iterations', 
                         fontsize=16, fontweight='bold')
            
            # Info box
            info_text = (
                f"Total Iterations: {num_iterations}\n"
                f"Samples per Iteration: {samples_per_iter:,}\n"
                f"Total Samples: {num_iterations * samples_per_iter:,}\n"
                f"Attempt Growth: +60 tokens/iteration\n"
                f"Max at iteration: 17 (1000 tokens)\n"
                f"Estimated Duration: {num_iterations:.1f} hours"
            )
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save the figure
            output_path = self.run_dir / 'timeline_preview.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 Timeline preview saved to: {output_path}")
            
            plt.close()
        except ImportError:
            print("⚠️  Matplotlib not available, skipping timeline visualization")
        except Exception as e:
            print(f"⚠️  Could not generate timeline preview: {e}")
    
    def show_timeline_analysis(self):
        """Show timeline analysis and attempt length progression with visualization"""
        num_iterations = self.state['config']['num_iterations']
        samples_per_epoch = self.state['config']['samples_per_epoch']
        epochs_per_iter = self.state['config']['initial_epochs']
        
        samples_per_iter = samples_per_epoch * epochs_per_iter
        
        print(f"\n{'='*70}")
        print("TRAINING TIMELINE ANALYSIS")
        print(f"{'='*70}")
        print(f"Total iterations: {num_iterations}")
        print(f"Samples per iteration: {samples_per_iter:,}")
        print(f"Total samples: {samples_per_iter * num_iterations:,}")
        print(f"Growth: Linear (+60 tokens per iteration)")
        print(f"\nATTEMPT LENGTH PROGRESSION:")
        print(f"{'Iter':<6} {'Total Samples':<15} {'Attempt Length':<15}")
        print("-" * 50)
        
        attempt_lengths = []
        tokens_per_iteration = 60
        for i in range(min(num_iterations, 25)):  # Show first 25 iterations
            total_samples = i * samples_per_iter
            
            if i == 0:
                attempt_len = 0
            else:
                # Linear growth: 60 tokens per iteration
                attempt_len = min(i * tokens_per_iteration, 1000)
            
            attempt_lengths.append(attempt_len)
            print(f"{i:<6} {total_samples:>14,} {attempt_len:>14}")
        
        # Find when we hit max
        max_iter = (1000 // tokens_per_iteration) + 1  # 1000/60 ≈ 17
        print(f"\n🎯 Maximum attempt length (1000) reached at iteration {max_iter}")
        print(f"{'='*70}\n")
        
        # Generate visualization
        self.generate_timeline_preview(num_iterations, samples_per_iter, attempt_lengths)
    
    def run_iteration(self, iteration: int) -> bool:
        """Run a complete iteration: training → rollout"""
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{self.state['config']['num_iterations'] - 1}")
        print(f"{'='*70}")
        
        # Dump iteration start
        self.dump_state("iteration_start", {
            "iteration": iteration,
            "total_iterations": self.state['config']['num_iterations'],
            "completed_phases": len(self.state.get('completed_phases', {}))
        }, iteration, "start")
        
        # Phase 1: Training (with trajectories from previous iteration if available)
        if not self.state_manager.is_phase_completed(iteration, 'training'):
            phase_name = "Training with trajectories" if iteration > 0 else "Initial training"
            print(f"\nPHASE 1: {phase_name}")
            checkpoint = self.run_training(iteration)
            if not checkpoint:
                print(f"❌ Training failed")
                return False
        else:
            checkpoint = self.state['completed_phases'][f'iter_{iteration}_training']['checkpoint']
            print(f"✓ Skipping training (already completed)")
            print(f"  Using checkpoint: {checkpoint}")
        
        # Phase 2: Generate trajectories (for next iteration to use)
        if not self.state_manager.is_phase_completed(iteration, 'trajectory_generation'):
            print(f"\nPHASE 2: Trajectory Generation")
            trajectory_dir = self.generate_trajectories(iteration, checkpoint)
            if not trajectory_dir:
                print(f"❌ Trajectory generation failed")
                return False
        else:
            print(f"✓ Skipping trajectory generation (already completed)")
        
        # Dump iteration completion
        self.dump_state("iteration_complete", {
            "iteration": iteration,
            "success": True,
            "final_checkpoint": checkpoint
        }, iteration, "complete")
        
        print(f"\n✓ Iteration {iteration} complete")
        return True
    
    def create_scheduler_config(self):
        """Create immutable scheduler configuration"""
        config = self.state['config']
        scheduler_config = {
            'warmup_epochs': config.get('warmup_epochs', 10),
            'total_epochs': max(config['initial_epochs'], config['mixed_epochs']),
            'batch_size': config['batch_size']
        }
        
        config_dir = self.run_dir / '_config'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'scheduler_config.json'
        
        with open(config_file, 'w') as f:
            json.dump(scheduler_config, f, indent=2)
        
        return config_file
    
    def run(self):
        """Main orchestration loop"""
        # Initialize state
        if self.args.resume:
            # Resume from existing run
            run_dir = Path(self.args.resume)
            self.state_manager = OrchestrationState(str(run_dir))
            self.state = self.state_manager.load()
            
            if not self.state:
                print(f"❌ Could not load state from {run_dir}")
                sys.exit(1)
            
            self.run_dir = Path(self.state['run_dir'])
            self.scheduler_config_path = self.run_dir / '_config' / 'scheduler_config.json'
            print(f"\nRESUMING TRAINING")
            print(f"Run: {self.run_dir}")
        else:
            # New run
            run_name = f"pseudo_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run_dir = Path('runs') / run_name
            self.run_dir.mkdir(parents=True, exist_ok=True)
            
            self.state_manager = OrchestrationState(str(self.run_dir))
            
            # Initialize state
            self.state = {
                'run_name': run_name,
                'run_dir': str(self.run_dir),
                'start_time': datetime.now().isoformat(),
                'config': {
                    'dataset': self.args.dataset,
                    'num_iterations': self.args.iterations,
                    'initial_epochs': self.args.initial_epochs,
                    'mixed_epochs': self.args.mixed_epochs,
                    'batch_size': self.args.batch_size,
                    'samples_per_epoch': self.args.samples_per_epoch,
                    'embed_size': self.args.embed_size,
                    'num_layers': self.args.num_layers,
                    'heads': self.args.heads,
                    'num_kv_heads': self.args.num_kv_heads,
                    'learning_rate': self.args.learning_rate,
                    'warmup_epochs': self.args.warmup_epochs,
                    'trajectory_samples': self.args.trajectory_samples,
                    'temperature': self.args.temperature,
                    'accumulation_steps': self.args.accumulation_steps,
                    'max_seq_length': self.args.max_seq_length
                },
                'checkpoints': {},
                'trajectories': {},
                'completed_phases': {},
                'current_iteration': 0
            }
            
            # Create scheduler config
            self.scheduler_config_path = self.create_scheduler_config()
            
            self.state_manager.save(self.state)
            print(f"\nSTARTING NEW TRAINING")
            print(f"Run: {self.run_dir}")
            
            # Show timeline analysis
            self.show_timeline_analysis()
        
        # Main iteration loop
        start_iter = self.state.get('current_iteration', 0)
        
        for iteration in range(start_iter, self.state['config']['num_iterations']):
            self.state['current_iteration'] = iteration
            self.state_manager.save(self.state)
            
            if not self.run_iteration(iteration):
                print(f"\n❌ Training failed at iteration {iteration}")
                return False
        
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETE")
        print(f"Final checkpoint: {self.state['checkpoints'].get('latest', 'unknown')}")
        print(f"{'='*70}")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Pseudo-RL Training Orchestrator')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, nargs='+', 
                       help='Paths to dataset files (required for new runs, not for resume)')
    
    # Training configuration
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of pseudo-RL iterations')
    parser.add_argument('--initial-epochs', type=int, default=100,
                       help='Epochs for initial training (iteration 0)')
    parser.add_argument('--mixed-epochs', type=int, default=100,
                       help='Epochs for training with trajectories')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--samples-per-epoch', type=int, default=None,
                       help='Samples per epoch (None = full dataset)')
    
    # Model configuration
    parser.add_argument('--embed-size', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--num-kv-heads', type=int, default=1)
    parser.add_argument('--max-seq-length', type=int, default=2700)
    
    # Optimizer configuration
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    
    # Trajectory configuration
    parser.add_argument('--trajectory-samples', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.8)
    
    # Orchestration
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from run directory')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.resume and not args.dataset:
        parser.error("--dataset is required for new runs (not needed for --resume)")
    
    # Confirmation
    if not args.yes and not args.resume:
        print("\nConfiguration:")
        print(f"  Datasets: {args.dataset}")
        print(f"  Iterations: {args.iterations}")
        print(f"  Initial epochs: {args.initial_epochs}")
        print(f"  Mixed epochs: {args.mixed_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Model: {args.embed_size}d, {args.num_layers}L, {args.heads}H")
        
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    orchestrator = FixedPseudoRLOrchestrator(args)
    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()