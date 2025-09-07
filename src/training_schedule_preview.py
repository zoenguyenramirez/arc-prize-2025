#!/usr/bin/env python3
"""
Generate training schedule preview and visualization for pseudo-RL training
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from src.schedulars import UnifiedTrainingScheduler


def generate_preview(duration_days, rollout_minutes, output_dir='temp'):
    """
    Generate and display training schedule preview
    
    Args:
        duration_days: Training duration in days
        rollout_minutes: Minutes between rollouts
        output_dir: Directory for output graphs
    """
    # Calculate parameters
    minutes_per_epoch = 5
    epochs_per_day = 288
    total_epochs = int(duration_days * epochs_per_day)
    samples_per_epoch = 400
    total_samples = total_epochs * samples_per_epoch
    
    # Calculate samples_per_double to reach max tokens at 20% of training
    # We need ~10 doublings to reach 1000 (2^10 = 1024)
    num_doublings = 10
    target_samples_for_max = int(total_samples * 0.2)  # Reach max at 20% of training
    samples_per_double = target_samples_for_max // num_doublings
    
    print(f"TRAINING SCHEDULE: {duration_days} days")
    print("=" * 70)
    print(f"Total epochs: {total_epochs:,}")
    print(f"Total samples: {total_samples:,}")
    print(f"Samples per doubling: {samples_per_double:,}")
    print("")
    
    # Create scheduler
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = UnifiedTrainingScheduler(
        optimizer=optimizer,
        run_dir='./runs/preview',
        warmup_epochs=min(100, int(total_epochs * 0.05)),
        total_epochs=total_epochs,
        samples_per_double=samples_per_double,
        batch_size=4
    )
    
    # Generate dry run
    epochs_per_rollout = rollout_minutes // minutes_per_epoch
    rollout_frequency_samples = epochs_per_rollout * samples_per_epoch
    rollout_time_minutes = 23  # ~23 minutes for 400 samples
    
    Path(output_dir).mkdir(exist_ok=True)
    graph_path = f'{output_dir}/{duration_days}day_schedule.png'
    
    result = scheduler.dry_run(
        dataset_size=400,
        batch_size=4,
        rollout_frequency_samples=rollout_frequency_samples,
        rollout_time_minutes=rollout_time_minutes,
        save_graphs=True,
        graph_path=graph_path
    )
    
    # Calculate statistics
    total_rollouts = len([r for r in result['rollout_epochs'] if r['epoch'] <= total_epochs])
    rollout_hours = (total_rollouts * rollout_time_minutes) / 60
    training_hours = duration_days * 24 - rollout_hours
    
    print("ATTEMPT LENGTH PROGRESSION")
    print("-" * 70)
    print(f"{'Day':<8} {'Hour':<8} {'Samples':<12} {'Attempt Length':<15}")
    print("-" * 70)
    for change in result['attempt_changes']:
        if change['epoch'] <= total_epochs:
            day = change['epoch'] / epochs_per_day
            hour = day * 24
            print(f"{day:<8.1f} {hour:<8.0f} {change['samples']:<12,} {change['new_length']:<15}")
    
    print("")
    print("ROLLOUT SCHEDULE")
    print("-" * 70)
    print(f"Total rollouts: {total_rollouts}")
    print(f"Frequency: Every {rollout_minutes} minutes")
    print(f"Rollout time: {rollout_hours:.1f} hours total ({rollout_hours/duration_days/24*100:.1f}%)")
    print(f"Training time: {training_hours:.1f} hours ({training_hours/duration_days/24*100:.1f}%)")
    
    print("")
    print("FINAL STATE")
    print("-" * 70)
    print(f"Final attempt length: {result['final_attempt_length']} tokens")
    print(f"Max memory per sample: ~{result['final_attempt_length'] * 2} MB")
    
    print("")
    print(f"📊 Visualization saved to: {graph_path}")
    print("=" * 70)
    
    return result


def generate_config(duration_days, warmup_epochs, batch_size, samples_per_epoch, output_path):
    """
    Generate scheduler configuration JSON file
    
    Args:
        duration_days: Training duration in days
        warmup_epochs: Number of warmup epochs
        batch_size: Batch size
        samples_per_epoch: Samples per epoch
        output_path: Path to save the config JSON
    """
    import json
    from pathlib import Path
    
    # Calculate total epochs and samples
    epochs_per_day = 288  # 5 minutes per epoch
    total_epochs = int(duration_days * epochs_per_day)
    total_samples = total_epochs * samples_per_epoch
    
    # Calculate samples_per_double to reach 1000 tokens at 20% of training
    # We need ~10 doublings to reach 1000 (2^10 = 1024)
    target_samples_for_max = int(total_samples * 0.2)
    samples_per_double = target_samples_for_max // 10
    
    config = {
        'warmup_epochs': warmup_epochs,
        'total_epochs': total_epochs,
        'samples_per_double': samples_per_double,
        'batch_size': batch_size
    }
    
    # Save config to JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Generated scheduler config at {output_path}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Samples per double: {samples_per_double}")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Generate training schedule preview and config')
    parser.add_argument('--days', type=float, default=7,
                       help='Training duration in days (default: 7)')
    parser.add_argument('--rollout-minutes', type=int, default=30,
                       help='Minutes between rollouts (default: 30)')
    parser.add_argument('--output-dir', type=str, default='temp',
                       help='Output directory for graphs (default: temp)')
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate scheduler config JSON')
    parser.add_argument('--config-output', type=str,
                       help='Path for scheduler config JSON')
    parser.add_argument('--warmup-epochs', type=int, default=100,
                       help='Number of warmup epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--samples-per-epoch', type=int, default=400,
                       help='Samples per epoch')
    
    args = parser.parse_args()
    
    # Generate preview (always)
    generate_preview(args.days, args.rollout_minutes, args.output_dir)
    
    # Also generate config if requested
    if args.generate_config:
        assert args.config_output, "--config-output required when using --generate-config"
        print("\n" + "=" * 70)
        print("CONFIG GENERATION")
        print("=" * 70)
        generate_config(
            args.days, 
            args.warmup_epochs,
            args.batch_size,
            args.samples_per_epoch,
            args.config_output
        )


if __name__ == '__main__':
    main()