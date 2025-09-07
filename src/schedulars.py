from torch.optim.lr_scheduler import _LRScheduler
import math
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class WarmupCosineLR(_LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):

        self.warmup_epochs = warmup_epochs

        self.total_epochs = total_epochs

        self.min_lr = min_lr

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:

            return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]

        else:

            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

            return [self.min_lr + (base_lr - self.min_lr) * 

                    (1 + math.cos(math.pi * progress)) / 2

                    for base_lr in self.base_lrs]


class UnifiedTrainingScheduler:
    """Unified scheduler for LR and attempt length with sample-based tracking"""
    
    def __init__(self, optimizer, run_dir, config_path=None, previous_state_path=None,
                 warmup_epochs=None, total_epochs=None, batch_size=None):
        """
        Initialize unified scheduler with immutable state management
        
        Args:
            optimizer: PyTorch optimizer
            run_dir: Directory to save new state (e.g., runs/pseudo_rl_XXX/iter0)
            config_path: Path to immutable scheduler_config.json
            previous_state_path: Path to previous scheduler_state.json (None for first iteration)
            warmup_epochs: Number of warmup epochs (fallback if no config)
            total_epochs: Total number of epochs (fallback if no config)
            batch_size: Batch size (fallback if no config)
        """
        import json
        import os
        
        self.optimizer = optimizer
        self.run_dir = Path(run_dir)
        
        # Save scheduler state in the training subdirectory to maintain immutability
        # Each training run gets its own scheduler state file
        self.state_save_dir = self.run_dir
        self.state_file = self.state_save_dir / "scheduler_state.json"
        
        # Load configuration from immutable config file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.warmup_epochs = config['warmup_epochs']
            self.total_epochs = config['total_epochs']
            self.batch_size = config['batch_size']
            
            print(f"Loaded scheduler config from {config_path}: warmup={self.warmup_epochs}, "
                  f"total_epochs={self.total_epochs}, batch_size={self.batch_size}")
        else:
            # Fallback to provided parameters (should not happen in orchestrated runs)
            self.warmup_epochs = warmup_epochs or 10
            self.total_epochs = total_epochs or 100
            self.batch_size = batch_size or 4
            if config_path:
                print(f"Warning: Config file not found at {config_path}, using fallbacks")
            else:
                print(f"Running without scheduler config, using parameters: "
                      f"warmup={self.warmup_epochs}, total_epochs={self.total_epochs}, "
                      f"batch_size={self.batch_size}")
        
        # Learning rate scheduler
        self.lr_scheduler = WarmupCosineLR(
            optimizer, 
            self.warmup_epochs, 
            self.total_epochs
        )
        
        # Attempt length configuration
        self.max_attempt_tokens = 1000  # 30x30 grid + overhead
        
        # Training state
        self.current_epoch = 0
        self.total_samples = 0  # Total samples processed
        self.total_steps = 0    # Total optimizer steps
        self.batches_in_epoch = 0  # Track batches per epoch
        self.samples_at_trajectory_start = None  # Samples when trajectories first became available
        self.current_iteration = 0  # Track which iteration we're in
        
        # Load previous state if provided (for continuing from previous iteration)
        if previous_state_path and os.path.exists(previous_state_path):
            with open(previous_state_path, 'r') as f:
                state = json.load(f)
            
            self._restore_state(state)
            
            # Load optimizer and lr_scheduler states from same directory as state file
            prev_dir = Path(previous_state_path).parent
            if 'optimizer_state_file' in state:
                # Look for the .pt files in the same directory as the state JSON
                optimizer_state_path = prev_dir / state['optimizer_state_file']
                if optimizer_state_path.exists():
                    self.optimizer.load_state_dict(torch.load(optimizer_state_path))
            
            if 'lr_scheduler_state_file' in state:
                lr_scheduler_state_path = prev_dir / state['lr_scheduler_state_file']
                if lr_scheduler_state_path.exists():
                    self.lr_scheduler.load_state_dict(torch.load(lr_scheduler_state_path))
            
            print(f"Loaded previous state from {previous_state_path}: epoch {self.current_epoch}, "
                  f"samples {self.total_samples}, attempt_length {self.get_current_attempt_length()}")
        elif previous_state_path:
            print(f"Warning: Previous state file not found at {previous_state_path}, starting fresh")
        else:
            # First iteration, no previous state
            print(f"Starting fresh scheduler state")
    
    def get_current_attempt_length(self):
        """Calculate current max attempt length based on iteration number"""
        # If no trajectories yet (iteration 0), always return 0
        if self.samples_at_trajectory_start is None:
            return 0
        
        # Linear growth: add 60 tokens per iteration
        # Iteration 0: no trajectories (0)
        # Iteration 1: first with trajectories (60)
        # Iteration 2: 120
        # Iteration 3: 180
        # ... etc up to max of 1000
        
        iterations_with_trajectories = self.current_iteration
        
        if iterations_with_trajectories <= 0:
            return 0
        
        # Linear growth: 60 tokens per iteration
        tokens_per_iteration = 60
        return min(iterations_with_trajectories * tokens_per_iteration, self.max_attempt_tokens)
    
    def step(self, batch_size=None):
        """
        Step the scheduler after each batch
        
        Args:
            batch_size: Size of current batch (uses self.batch_size if not provided)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        self.total_steps += 1
        self.total_samples += batch_size
        self.batches_in_epoch += 1
    
    def epoch_step(self):
        """Called at end of epoch - steps LR scheduler and saves state"""
        self.lr_scheduler.step()
        self.current_epoch += 1
        self.batches_in_epoch = 0
        self.save_state()
    
    def should_rollout(self, rollout_frequency_samples=None):
        """
        Check if it's time for rollout generation
        
        Args:
            rollout_frequency_samples: Samples between rollouts (None means no rollouts)
        """
        if rollout_frequency_samples is None:
            return False
        return (self.total_samples % rollout_frequency_samples == 0 and 
                self.total_samples > 0)
    
    def save_state(self):
        """Save all scheduler state to disk in JSON format"""
        # Save non-serializable parts as .pt files
        lr_scheduler_file = self.state_save_dir / "lr_scheduler_state.pt"
        optimizer_file = self.state_save_dir / "optimizer_state.pt"
        
        # Create directory if needed
        self.state_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch states separately
        torch.save(self.lr_scheduler.state_dict(), lr_scheduler_file)
        torch.save(self.optimizer.state_dict(), optimizer_file)
        
        # Save JSON state
        state = {
            'current_epoch': self.current_epoch,
            'total_steps': self.total_steps,
            'total_samples': self.total_samples,
            'batch_size': self.batch_size,
            'batches_in_epoch': self.batches_in_epoch,
            'attempt_length': self.get_current_attempt_length(),
            'samples_at_trajectory_start': self.samples_at_trajectory_start,
            'current_iteration': self.current_iteration,
            'timestamp': datetime.now().isoformat(),
            'lr_scheduler_state_file': 'lr_scheduler_state.pt',
            'optimizer_state_file': 'optimizer_state.pt'
        }
        
        # Write to temp file first for atomicity
        temp_file = self.state_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Atomically move to final location
        temp_file.replace(self.state_file)
    
    def load_state(self):
        """Load scheduler state from disk"""
        if not self.state_file.exists():
            # Try legacy .pt file
            legacy_file = self.run_dir / "scheduler_state.pt"
            if legacy_file.exists():
                print(f"Loading legacy state from {legacy_file}")
                state = torch.load(legacy_file, map_location='cpu')
                self._restore_state(state)
                # Save as JSON for next time
                self.save_state()
                return
            print(f"No scheduler state found at {self.state_file}")
            return
            
        # Load JSON state
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self._restore_state(state)
    
    def _restore_state(self, state):
        """Restore scheduler state from a loaded state dict"""
        # Handle both legacy .pt format and new JSON format
        if 'lr_scheduler_state' in state:
            # Legacy format: everything in one dict
            self.lr_scheduler.load_state_dict(state['lr_scheduler_state'])
            if 'optimizer_state' in state:
                self.optimizer.load_state_dict(state['optimizer_state'])
        elif 'lr_scheduler_state_file' in state:
            # New format: separate files referenced from JSON
            # Files should be in same directory as the state file that was loaded
            # Don't look in self.run_dir as that's the NEW training dir
            pass  # Already loaded in __init__ from previous_state_path's directory
        
        # Restore scalar values
        self.current_epoch = state.get('current_epoch', 0)
        self.total_steps = state.get('total_steps', 0)
        self.total_samples = state.get('total_samples', 0)
        self.batch_size = state.get('batch_size', 4)
        self.batches_in_epoch = state.get('batches_in_epoch', 0)
        self.samples_at_trajectory_start = state.get('samples_at_trajectory_start', None)
        self.current_iteration = state.get('current_iteration', 0)
        
        print(f"Resumed from epoch {self.current_epoch}, "
              f"{self.total_samples:,} samples, "
              f"attempt_length={self.get_current_attempt_length()}")
    
    def get_lr(self):
        """Get current learning rate"""
        return self.lr_scheduler.get_last_lr()[0]
    
    def get_status(self):
        """Get current scheduler status for logging"""
        return {
            'epoch': self.current_epoch,
            'lr': self.get_lr(),
            'attempt_length': self.get_current_attempt_length(),
            'total_samples': self.total_samples,
            'total_steps': self.total_steps,
            'batches_in_epoch': self.batches_in_epoch
        }
    
    def __repr__(self):
        """String representation of scheduler state"""
        status = self.get_status()
        return (f"UnifiedTrainingScheduler("
                f"epoch={status['epoch']}, "
                f"lr={status['lr']:.6f}, "
                f"attempt={status['attempt_length']}, "
                f"samples={status['total_samples']:,})")
    
    def dry_run(self, dataset_size, batch_size=None, rollout_frequency_samples=None, 
                rollout_time_minutes=30, save_graphs=True, graph_path='scheduler_preview.png'):
        """
        Perform a dry run to show training schedule and milestones
        
        Args:
            dataset_size: Number of samples in dataset
            batch_size: Batch size (uses self.batch_size if not provided)
            rollout_frequency_samples: How often to do rollouts (None = no rollouts)
            rollout_time_minutes: Estimated time for each rollout
            save_graphs: Whether to save visualization graphs
            graph_path: Where to save the graphs
            
        Returns:
            Schedule information dict
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        samples_per_epoch = dataset_size
        batches_per_epoch = dataset_size // batch_size
        
        print("\n" + "=" * 70)
        print("TRAINING SCHEDULE DRY RUN")
        print("=" * 70)
        print(f"Dataset size: {dataset_size} samples")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {batches_per_epoch}")
        print(f"Samples per epoch: {samples_per_epoch}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Attempt length: Linear growth of 60 tokens per iteration")
        
        if rollout_frequency_samples:
            print(f"Rollout frequency: Every {rollout_frequency_samples:,} samples")
            print(f"Estimated rollout time: {rollout_time_minutes} minutes")
        else:
            print("Rollouts: Disabled")
        
        print("\n" + "-" * 70)
        print("ATTEMPT LENGTH PROGRESSION")
        print("-" * 70)
        
        # Simulate training
        milestones = []
        rollout_epochs = []
        attempt_changes = []
        
        current_samples = 0
        last_attempt_length = 0
        
        for epoch in range(self.total_epochs):
            epoch_start_samples = current_samples
            current_samples += samples_per_epoch
            
            # Calculate attempt length at this epoch (iteration-based linear growth)
            # For dry run, we simulate based on epoch as iteration proxy
            # In actual training, this is based on self.current_iteration
            if epoch == 0:
                attempt_length = 0  # No trajectories in first iteration
            else:
                # Linear growth: 60 tokens per iteration
                tokens_per_iteration = 60
                attempt_length = min(epoch * tokens_per_iteration, self.max_attempt_tokens)
            
            # Check for attempt length change
            if attempt_length != last_attempt_length:
                attempt_changes.append({
                    'epoch': epoch,
                    'samples': epoch_start_samples,
                    'old_length': last_attempt_length,
                    'new_length': attempt_length
                })
                last_attempt_length = attempt_length
            
            # Check for rollouts during this epoch
            if rollout_frequency_samples:
                # Check each sample in the epoch
                for sample in range(epoch_start_samples, current_samples, batch_size):
                    if sample > 0 and sample % rollout_frequency_samples == 0:
                        rollout_epochs.append({
                            'epoch': epoch,
                            'samples': sample,
                            'attempt_length': attempt_length
                        })
        
        # Print attempt length changes
        print(f"{'Epoch':<8} {'Samples':<12} {'Attempt Length':<15} {'Change'}")
        print("-" * 50)
        
        for change in attempt_changes:
            if change['epoch'] == 0:
                print(f"{change['epoch']:<8} {change['samples']:<12,} {change['new_length']:<15} (start)")
            else:
                print(f"{change['epoch']:<8} {change['samples']:<12,} {change['new_length']:<15} "
                      f"({change['old_length']} → {change['new_length']})")
        
        # Print rollout schedule
        if rollout_epochs:
            print("\n" + "-" * 70)
            print("ROLLOUT SCHEDULE")
            print("-" * 70)
            print(f"{'Rollout #':<12} {'Epoch':<8} {'Samples':<12} {'Attempt Len'}")
            print("-" * 50)
            
            for i, rollout in enumerate(rollout_epochs[:20], 1):  # Show first 20
                print(f"{i:<12} {rollout['epoch']:<8} {rollout['samples']:<12,} {rollout['attempt_length']}")
            
            if len(rollout_epochs) > 20:
                print(f"... and {len(rollout_epochs) - 20} more rollouts")
            
            total_rollout_time = len(rollout_epochs) * rollout_time_minutes
            print(f"\nTotal rollout time: {total_rollout_time} minutes ({total_rollout_time/60:.1f} hours)")
        
        # Calculate memory implications
        print("\n" + "-" * 70)
        print("MEMORY IMPLICATIONS")
        print("-" * 70)
        
        base_memory_mb = 100  # Base memory for model
        memory_per_token_mb = 0.5  # Estimated memory per attempt token
        
        print(f"{'Attempt Length':<15} {'Extra Memory (MB)':<20} {'Total (MB)'}")
        print("-" * 50)
        
        for length in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]:
            extra_memory = length * memory_per_token_mb * batch_size
            total_memory = base_memory_mb + extra_memory
            if any(c['new_length'] == length for c in attempt_changes):
                epoch = next(c['epoch'] for c in attempt_changes if c['new_length'] == length)
                print(f"{length:<15} {extra_memory:<20.1f} {total_memory:<10.1f} (epoch {epoch})")
        
        # Summary statistics
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        
        total_samples = self.total_epochs * samples_per_epoch
        print(f"Total samples to process: {total_samples:,}")
        # Final attempt length based on iterations
        final_attempt = min(self.total_epochs * 60, self.max_attempt_tokens)
        print(f"Final attempt length: {final_attempt} tokens")
        print(f"Number of attempt length increases: {len(attempt_changes)}")
        
        if rollout_epochs:
            print(f"Total number of rollouts: {len(rollout_epochs)}")
            print(f"Samples between rollouts: {rollout_frequency_samples:,}")
        
        # Estimate time
        seconds_per_sample = 0.1  # Rough estimate
        training_time_hours = (total_samples * seconds_per_sample) / 3600
        
        if rollout_epochs:
            total_time_hours = training_time_hours + (total_rollout_time / 60)
            print(f"\nEstimated training time: {training_time_hours:.1f} hours")
            print(f"Estimated rollout time: {total_rollout_time/60:.1f} hours")
            print(f"Estimated total time: {total_time_hours:.1f} hours")
        else:
            print(f"\nEstimated training time: {training_time_hours:.1f} hours")
        
        print("=" * 70)
        
        # Generate graphs if requested
        if save_graphs:
            self._generate_training_graphs(
                dataset_size=dataset_size,
                batch_size=batch_size,
                attempt_changes=attempt_changes,
                rollout_epochs=rollout_epochs,
                save_path=graph_path,
                rollout_time_minutes=rollout_time_minutes
            )
            print(f"\n📊 Graphs saved to: {graph_path}")
        
        return {
            'dataset_size': dataset_size,
            'batch_size': batch_size,
            'total_epochs': self.total_epochs,
            'total_samples': total_samples,
            'attempt_changes': attempt_changes,
            'rollout_epochs': rollout_epochs,
            'final_attempt_length': min(self.total_epochs * 60, self.max_attempt_tokens)
        }
    
    def _generate_training_graphs(self, dataset_size, batch_size, attempt_changes, 
                                  rollout_epochs, save_path='scheduler_preview.png', rollout_time_minutes=30):
        """
        Generate visualization graphs for the training schedule
        
        Args:
            dataset_size: Number of samples in dataset
            batch_size: Batch size
            attempt_changes: List of attempt length changes
            rollout_epochs: List of rollout points
            save_path: Where to save the graph
        """
        samples_per_epoch = dataset_size
        
        # Simulate full training to get all data points
        epochs = []
        samples = []
        attempt_lengths = []
        learning_rates = []
        
        current_samples = 0
        for epoch in range(self.total_epochs):
            epochs.append(epoch)
            samples.append(current_samples)
            
            # Calculate attempt length at this epoch (iteration-based linear growth)
            if epoch == 0:
                attempt_length = 0  # No trajectories in first iteration
            else:
                # Linear growth: 60 tokens per iteration
                tokens_per_iteration = 60
                attempt_length = min(epoch * tokens_per_iteration, self.max_attempt_tokens)
            
            attempt_lengths.append(attempt_length)
            
            # Calculate learning rate (simulate warmup and cosine)
            if epoch < self.warmup_epochs:
                lr = (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = 0.01 + (1 - 0.01) * (1 + math.cos(math.pi * progress)) / 2
            learning_rates.append(lr)
            
            current_samples += samples_per_epoch
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Schedule Preview - {self.total_epochs} Epochs, {dataset_size} Samples/Epoch', 
                     fontsize=16)
        
        # Plot 1: Attempt Length Progression
        ax1 = axes[0, 0]
        ax1.plot(epochs, attempt_lengths, 'b-', linewidth=2)
        ax1.fill_between(epochs, attempt_lengths, alpha=0.3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Max Attempt Length (tokens)')
        ax1.set_title('Attempt Length Progression (Exponential Growth)')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('symlog')  # Log scale for better visualization
        
        # Add change markers
        for change in attempt_changes:
            ax1.axvline(x=change['epoch'], color='r', linestyle='--', alpha=0.5)
            ax1.text(change['epoch'], change['new_length'], 
                    f"→{change['new_length']}", fontsize=8, ha='left')
        
        # Plot 2: Learning Rate Schedule
        ax2 = axes[0, 1]
        ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
        ax2.fill_between(epochs, learning_rates, alpha=0.3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate (normalized)')
        ax2.set_title(f'Learning Rate Schedule (Warmup: {self.warmup_epochs} epochs)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.warmup_epochs, color='orange', linestyle='--', 
                   alpha=0.5, label='End of Warmup')
        ax2.legend()
        
        # Plot 3: Samples vs Attempt Length
        ax3 = axes[1, 0]
        ax3.plot(samples, attempt_lengths, 'purple', linewidth=2)
        ax3.fill_between(samples, attempt_lengths, alpha=0.3)
        ax3.set_xlabel('Total Samples Processed')
        ax3.set_ylabel('Max Attempt Length (tokens)')
        ax3.set_title(f'Attempt Length vs Samples (Linear growth: 60 tokens per iteration)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('linear')
        ax3.set_yscale('symlog')
        
        # Add iteration markers (every epoch = iteration in dry run)
        for i in range(1, min(12, self.total_epochs)):
            x_pos = i * samples_per_epoch  # Each epoch represents an iteration
            if x_pos <= max(samples):
                ax3.axvline(x=x_pos, color='gray', linestyle=':', alpha=0.3)
        
        # Plot 4: Rollout Schedule Timeline
        ax4 = axes[1, 1]
        
        if rollout_epochs:
            # Create a more readable timeline showing rollout points
            rollout_epochs_list = rollout_epochs[:20]  # Show first 20
            rollout_times = [r['epoch'] for r in rollout_epochs_list]
            rollout_attempts = [r['attempt_length'] for r in rollout_epochs_list]
            
            # Create scatter plot for rollout points
            ax4.scatter(rollout_times, rollout_attempts, c='red', s=100, alpha=0.7, 
                       marker='v', label='Rollout Points', zorder=5)
            
            # Add background line showing attempt progression
            ax4.plot(epochs, attempt_lengths, 'b-', alpha=0.3, linewidth=1, 
                    label='Attempt Length')
            
            # Add vertical lines at rollout points
            for rollout_time in rollout_times:
                ax4.axvline(x=rollout_time, color='red', linestyle=':', alpha=0.2)
            
            # Annotate first few rollouts
            for i, (time, attempt) in enumerate(zip(rollout_times[:5], rollout_attempts[:5])):
                ax4.annotate(f'R{i+1}', xy=(time, attempt), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Attempt Length (tokens)')
            ax4.set_title(f'Rollout Schedule ({len(rollout_epochs)} rollouts total)')
            ax4.set_yscale('symlog')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left')
            
            # Add text summary
            rollout_text = f"Rollouts: Every {epochs[1] * (rollout_epochs[0]['epoch'] if rollout_epochs else 1)} epochs\n"
            rollout_text += f"Total: {len(rollout_epochs)} rollouts\n"
            rollout_text += f"Time: {len(rollout_epochs) * rollout_time_minutes} min total"
            ax4.text(0.98, 0.02, rollout_text, transform=ax4.transAxes,
                    fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # No rollouts scheduled
            ax4.text(0.5, 0.5, 'No Rollouts Scheduled', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=14, alpha=0.5)
            ax4.set_title('Rollout Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Attempt Length (tokens)')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nGraphs saved to: {save_path}")

