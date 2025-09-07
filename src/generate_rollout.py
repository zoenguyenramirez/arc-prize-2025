#!/usr/bin/env python3
"""
Standalone rollout script for generating trajectories
Supports dry-run to preview what will be generated
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime
import logging
import sys
import os

from src.rl_trajectory_generator import TrajectoryGenerator
from src.utils.logger_helper import setup_logging
from src.utils.helper import set_deterministic


def dry_run(args):
    """
    Perform a dry run to show what will be generated
    
    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 70)
    print("ROLLOUT DRY RUN")
    print("=" * 70)
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return False
    else:
        print(f"✓ Checkpoint found: {args.checkpoint}")
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return False
    else:
        # Load dataset to get size
        dataset = torch.load(args.dataset, weights_only=False)
        if isinstance(dataset, dict):
            dataset_size = len(dataset.get('dataset', dataset.get('data', [])))
        elif isinstance(dataset, list):
            dataset_size = len(dataset)
        else:
            dataset_size = 0
        print(f"✓ Dataset found: {dataset_size} samples")
    
    # Calculate what will be generated
    num_samples = min(args.num_samples, dataset_size) if args.num_samples else dataset_size
    num_iterations = args.num_iterations
    batch_size = 1  # Always use batch_size=1
    
    print(f"\n" + "-" * 70)
    print("GENERATION PLAN")
    print("-" * 70)
    print(f"Samples to process: {num_samples}")
    print(f"Refinement iterations: {num_iterations}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens per sample: {args.max_tokens}")
    print(f"Batch size: {batch_size} (fixed)")
    
    # Output location
    output_dir = Path(args.output_dir)
    print(f"\n" + "-" * 70)
    print("OUTPUT LOCATION")
    print("-" * 70)
    print(f"Output directory: {output_dir}")
    
    if output_dir.exists():
        existing_files = list(output_dir.glob("*.pth"))
        print(f"⚠️  Directory exists with {len(existing_files)} existing files")
        if not args.force:
            print("   Use --force to overwrite existing files")
    else:
        print("✓ Will create new directory")
    
    # Time estimation
    seconds_per_sample = 2.0  # Rough estimate
    seconds_per_iteration = 1.0  # Additional time per refinement
    total_time = num_samples * (seconds_per_sample + num_iterations * seconds_per_iteration)
    
    print(f"\n" + "-" * 70)
    print("TIME ESTIMATION")
    print("-" * 70)
    print(f"Estimated time per sample: {seconds_per_sample + num_iterations * seconds_per_iteration:.1f} seconds")
    print(f"Total estimated time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
    
    # Memory estimation
    tokens_per_sample = args.max_tokens
    memory_per_token = 0.01  # MB (rough estimate)
    memory_per_sample = tokens_per_sample * memory_per_token
    peak_memory = memory_per_sample * 10  # Keep some samples in memory
    
    print(f"\n" + "-" * 70)
    print("MEMORY ESTIMATION")
    print("-" * 70)
    print(f"Max tokens per sample: {tokens_per_sample}")
    print(f"Estimated memory per sample: {memory_per_sample:.1f} MB")
    print(f"Estimated peak memory: {peak_memory:.1f} MB")
    
    # Summary
    print(f"\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    
    if num_iterations == 0:
        print(f"Will generate {num_samples} trajectories WITHOUT refinement")
    else:
        print(f"Will generate {num_samples} trajectories with {num_iterations} refinement iterations each")
    
    print(f"Output will be saved to: {output_dir}")
    
    # Expected output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    expected_file = output_dir / f"trajectories_{timestamp}.pt"
    print(f"Expected output file: {expected_file}")
    
    print("=" * 70)
    
    return True


def main():
    """Main entry point for rollout generation"""
    parser = argparse.ArgumentParser(description='Generate trajectories for pseudo-RL training')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, nargs='+', required=True,
                       help='Path(s) to dataset file(s) (.pth)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for trajectories')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Optional explicit output filename (for iteration-specific naming)')
    
    # Generation parameters
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to process (default: all)')
    parser.add_argument('--num-iterations', type=int, default=0,
                       help='Number of refinement iterations (default: 0 = no refinement)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (0=deterministic)')
    parser.add_argument('--max-tokens', type=int, default=900,
                       help='Maximum tokens to generate per sample')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what will be generated without actually running')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing output directory')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-file', type=str, default='rollout_generation.log',
                       help='Log file path')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile optimization')
    
    args = parser.parse_args()
    
    # If dry-run, just show what would happen
    if args.dry_run:
        success = dry_run(args)
        if success:
            print("\n✅ Dry run complete. Use without --dry-run to actually generate trajectories.")
        else:
            print("\n❌ Dry run failed. Please fix the issues above.")
        return 0 if success else 1
    
    # Setup for actual generation
    setup_logging(args.log_file)
    if args.seed:
        set_deterministic(args.seed)
    
    # Create output directory (always allow existing directories since we use unique filenames)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Load dataset(s)
    logging.info(f"Loading dataset from {args.dataset}...")
    
    # Handle multiple dataset files
    if len(args.dataset) == 1:
        # Single dataset file
        dataset = torch.load(args.dataset[0], weights_only=False)
    else:
        # Multiple dataset files - load and combine like train.py does
        from src.prepare_data import load_datasets
        dataset, _, _ = load_datasets(args.dataset)
    
    # Extract actual data from dataset structure
    if isinstance(dataset, dict):
        if 'dataset' in dataset:
            dataset = dataset['dataset']
        elif 'data' in dataset:
            dataset = dataset['data']
    
    # Don't truncate here - let TrajectoryGenerator do random sampling
    # This ensures we sample randomly from the full dataset, not just the first N samples
    full_dataset_size = len(dataset)
    samples_to_process = args.num_samples if args.num_samples else full_dataset_size
    
    logging.info(f"Full dataset size: {full_dataset_size} samples")
    logging.info(f"Will randomly sample {samples_to_process} samples")
    
    # Initialize trajectory generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    generator = TrajectoryGenerator(
        checkpoint_path=args.checkpoint,
        device=device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_compile=not args.no_compile,
    )
    
    # TODO: Add support for iterative refinement (num_iterations > 0)
    # This requires modifying TrajectoryGenerator to:
    # 1. Generate initial trajectory
    # 2. Feed it back with ATTEMPT_START/END tokens
    # 3. Generate refined output
    # 4. Repeat for num_iterations
    if args.num_iterations > 0:
        logging.warning("Iterative refinement not yet implemented in TrajectoryGenerator")
        logging.warning("Generating single-pass trajectories only")
        logging.warning(f"Requested {args.num_iterations} iterations, but only doing 1")
    
    # Generate trajectories
    logging.info("Starting trajectory generation...")
    start_time = datetime.now()
    
    # TODO: Verify that TrajectoryGenerator stores ground_truth_output field
    # This is needed for the pseudo-RL training to know the correct answer
    output_path = generator.process_dataset(
        dataset=dataset,
        output_dir=str(output_dir),
        output_file=args.output_file,  # Pass iteration-specific filename if provided
        num_batches=samples_to_process  # Pass the number of samples to randomly select
    )
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logging.info("=" * 70)
    logging.info("GENERATION COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Output saved to: {output_path}")
    logging.info(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logging.info(f"Average time per sample: {duration/samples_to_process:.2f} seconds")
    
    # Print statistics
    generator.print_statistics()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())