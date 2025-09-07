"""
Simplified RL Trajectory Generator for Experience Replay
Only stores trajectories - no logits, no rewards, minimal memory usage
"""

import torch
from torch.amp import autocast
from typing import List, Dict, Optional, Any
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
import argparse

from src.fast_inference_kv import FastBatchInferenceKV
from src.token import SpecialToken
from src.load_data import load_from_json, GridDataset
from src.utils.logger_helper import setup_logging
from src.utils.helper import set_deterministic
from src.inference_output_saver import InferenceOutputSaver


class TrajectoryGenerator:
    """
    Simplified trajectory generator for RL experience replay
    Generates sequences up to 900 tokens and stores them efficiently
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        max_tokens: int = 900,
        temperature: float = 0.0,  # Default to 0 for deterministic
        use_compile: bool = True,
    ):
        """
        Initialize trajectory generator
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Compute device (cuda/cpu)
            max_tokens: Maximum tokens to generate (default 900)
            temperature: Sampling temperature (0=deterministic, default 0)
            use_compile: Whether to use torch.compile
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.batch_size = 1  # Always use batch_size=1 for consistency
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize inference engine with REAL KV caching
        self.inference_engine = FastBatchInferenceKV(
            checkpoint_path=checkpoint_path,
            device=device,
            use_compile=use_compile
        )
        
        # Simple statistics
        self.total_trajectories = 0
        self.total_time = 0.0
        
        logging.info(f"Trajectory Generator initialized")
        logging.info(f"Device: {device}, Batch size: 1, Max tokens: {max_tokens}, Temperature: {temperature}")
    
    @torch.no_grad()
    def generate_batch(
        self,
        input_sequences: List[List],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectories for experience replay
        
        Args:
            input_sequences: List of input sequences (prompts)
            
        Returns:
            Dictionary containing:
                - trajectories: Generated sequences
                - finished: Whether each sequence hit END token
        """
        start_time = time.time()
        
        # Generate trajectories (currently always deterministic - uses argmax)
        results = self.inference_engine.generate_batch(
            input_sequences=input_sequences,
            max_new_tokens=self.max_tokens,
            return_logits=False,  # Don't save logits - saves memory
            pad_token_id=SpecialToken.PAD.value,
            eos_token_id=SpecialToken.END.value,
        )
        
        # Check which sequences finished (reached END token)
        batch_size = len(input_sequences)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            seq_tokens = results['tokens'][i]
            # Check if END token appears
            end_positions = (seq_tokens == SpecialToken.END.value).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                finished[i] = True
        
        # Update statistics
        self.total_trajectories += batch_size
        self.total_time += time.time() - start_time
        
        # Return only what's needed for experience replay
        return {
            'trajectories': results['sequences'],  # Full sequences with coordinates
            'finished': finished,  # Which sequences completed
        }
    
    def process_dataset(
        self,
        dataset: Any,
        num_batches: Optional[int] = None,
        output_dir: str = "rl_trajectories",
        output_file: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> str:
        """
        Process dataset and save trajectories for experience replay
        
        Args:
            dataset: Dataset to process
            num_batches: Number of batches to process (None for all)
            output_dir: Directory to save trajectories
            output_file: Optional explicit output filename (overrides default naming)
            random_seed: Seed for random sampling (None for non-deterministic)
            
        Returns:
            Path to saved trajectory file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import remove_last_grid for ground truth extraction
        from src.utils.grid_data_process import remove_last_grid
        
        # Create output file path and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_file is None:
            output_file = str(output_dir / f"trajectories_{timestamp}.pt")
        # else use the provided output_file path as-is
        
        # Use InferenceOutputSaver for consistency with src.sample
        output_saver = InferenceOutputSaver(output_file)
        
        all_trajectories = []  # Keep for backward compatibility
        ground_truth_outputs = []  # Store ground truth for each sample
        
        # Process dataset one sample at a time (batch_size=1)
        num_samples = min(len(dataset), num_batches) if num_batches else len(dataset)
        
        # Randomly sample indices from the entire dataset
        assert num_samples <= len(dataset), f"Cannot sample {num_samples} from dataset of size {len(dataset)}"
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            logging.info(f"Using random seed {random_seed} for sampling")
        
        # Sample without replacement using randperm
        sampled_indices = torch.randperm(len(dataset))[:num_samples]
        sampled_indices = sampled_indices.tolist()
        logging.info(f"Randomly sampled {num_samples} indices from dataset of size {len(dataset)}")
        
        for i, idx in enumerate(sampled_indices):
            # Get the raw compact data before tokenization
            compact_data = dataset.convert_to_int_lists(dataset.data[idx])
            
            # Extract ground truth by removing the last grid from compact format
            input_without_output_compact = remove_last_grid(compact_data)
            
            # Extract ground truth output (the part that was removed)
            ground_truth_output_compact = compact_data[len(input_without_output_compact):]
            
            # Now tokenize the input for generation
            from src.utils.grid_data_process import tokenize_compact_task
            input_tokenized = tokenize_compact_task(input_without_output_compact)
            input_seq = [input_tokenized]  # Wrap in list for batch processing
            
            # Also tokenize the ground truth for storage
            # ground_truth_output_compact contains only the output part, so use single_output=True
            ground_truth_tokenized = tokenize_compact_task(ground_truth_output_compact, single_output=True)
            ground_truth_outputs.append(ground_truth_tokenized)
            
            # Generate trajectory for this single sample
            print(f"\rProcessing sample {i + 1}/{num_samples} (dataset idx: {idx})...", flush=True, end="")
            
            results = self.generate_batch(input_seq)
            
            # Extract the generated sequence (squeeze batch dimension)
            trajectory = results['trajectories'].squeeze(0)  # Remove batch dim
            finished = results['finished'].item()  # Convert to bool
            
            # Store trajectory for backward compatibility
            all_trajectories.append(trajectory)
            
            # Add to output saver (it will handle input/output separation)
            # Convert trajectory to list format for compatibility
            generated_sample = trajectory.tolist() if torch.is_tensor(trajectory) else trajectory
            
            # Update input_sequence to use the tokenized input (without ground truth)
            input_sequence_trimmed = {'task': input_tokenized}
            
            output_saver.add_sample(
                generated_sample=generated_sample,
                input_sequence=input_sequence_trimmed,
                finished=finished,
                sample_index=idx,
                generated_length=len(trajectory) - len(input_tokenized),
                matches=finished  # For trajectory generator, finished means successful
            )
        
        print()  # New line after progress
        
        # For pseudo-RL, we need a different format - save as list of trajectory dicts
        trajectories_for_rl = []
        for i in range(num_samples):
            trajectory_dict = {
                'task_id': f"task_{sampled_indices[i]}",
                'trajectory_id': f"{timestamp}_{i}",
                'input_sequence': output_saver.input_sequences[i] if i < len(output_saver.input_sequences) else [],
                'generated_output': output_saver.generated_sequences[i] if i < len(output_saver.generated_sequences) else [],
                'ground_truth_output': ground_truth_outputs[i] if i < len(ground_truth_outputs) else [],
                'timestamp': timestamp,
                'temperature': self.temperature,
                'finished': output_saver.finished[i] if i < len(output_saver.finished) else False,
            }
            trajectories_for_rl.append(trajectory_dict)
        
        # Save in the format expected by trajectory_loader
        output_path = Path(output_file) if output_file else Path(output_dir) / f"trajectories_{timestamp}.pt"
        torch.save(trajectories_for_rl, output_path)
        
        # Log save info
        logging.info('\nSaved output to: %s', output_path)
        logging.info('  - Generated sequences: %d', len(trajectories_for_rl))
        if output_path.exists():
            logging.info('  - File size: %.2f MB', output_path.stat().st_size / (1024 * 1024))
        
        # Print statistics
        if self.total_time > 0:
            logging.info(f"Generation rate: {self.total_trajectories / self.total_time:.1f} trajectories/sec")
        
        return str(output_path)
    
    def print_statistics(self):
        """Print generation statistics"""
        print(f"\nTrajectory Generation Statistics:")
        print(f"  Total trajectories: {self.total_trajectories}")
        print(f"  Total time: {self.total_time:.1f}s")
        if self.total_time > 0:
            print(f"  Rate: {self.total_trajectories / self.total_time:.1f} trajectories/sec")


def main():
    """Main entry point for trajectory generation"""
    parser = argparse.ArgumentParser(description='RL Trajectory Generator for Experience Replay')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data-source', type=str, default='arc-agi_training',
                       help='Data source to use')
    parser.add_argument('--second-only', action='store_true', default=False,
                       help='Use second test only (for comparison with src.sample)')
    parser.add_argument('--num-batches', type=int, default=None,
                       help='Number of batches to process (default: all)')
    
    # Generation arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for generation')
    parser.add_argument('--max-tokens', type=int, default=900,
                       help='Maximum tokens to generate per sequence')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='rl_trajectories',
                       help='Output directory for trajectories')
    
    # Other arguments
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile optimization')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-file', type=str, default='trajectory_generation.log',
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_file)
    if args.seed:
        set_deterministic(args.seed)
    
    # Load data - need both challenges and solutions for ground truth extraction
    logging.info(f"Loading data from {args.data_source}...")
    challenges, solutions = load_from_json(args.data_source, './input_data/')
    
    # Create dataset with second_only flag - pass solutions for ground truth
    dataset = GridDataset.load_from_paired_file(challenges, solutions, second_only=args.second_only)
    logging.info(f"Loaded {len(dataset)} samples (second_only={args.second_only})")
    
    # Initialize generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TrajectoryGenerator(
        checkpoint_path=args.checkpoint,
        device=device,
        max_tokens=args.max_tokens,
        use_compile=not args.no_compile,
    )
    
    # Process dataset with random sampling
    logging.info("Starting trajectory generation with random sampling...")
    output_path = generator.process_dataset(
        dataset=dataset,
        num_batches=args.num_batches,
        output_dir=args.output_dir,
        random_seed=args.seed,  # Use the seed for reproducible random sampling
    )
    
    # Print final statistics
    generator.print_statistics()
    
    logging.info(f"Trajectory generation complete. Output: {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()