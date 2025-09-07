"""
Collects and saves inference outputs to .pt files
"""
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import hashlib


class InferenceOutputSaver:
    """Collects inference outputs on the fly and saves to .pt file"""
    
    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize the saver
        
        Args:
            output_file: Output file path (uses tmp/ if not provided)
        """
        # Set default output file if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'tmp/sample_output_{timestamp}.pt'
            logging.info(f'No output file specified, using: {output_file}')
        
        self.output_file = output_file
        self.generated_sequences = []
        self.input_sequences = []
        self.input_hashes = []  # 64-bit hashes for fast lookup
        self.finished = []
        self.match_info = []
    
    def add_sample(
        self,
        generated_sample: list,
        input_sequence: dict,
        finished: bool,
        sample_index: int,
        generated_length: int,
        matches: bool
    ):
        """
        Add a single sample to the collection
        
        Args:
            generated_sample: Full generated sequence (including input)
            input_sequence: Input sequence dict with 'task' field
            finished: Whether sequence finished successfully
            sample_index: Index of the sample
            generated_length: Number of tokens generated
            matches: Whether it matches expected
        """
        # Extract input and output portions
        input_task = input_sequence['task']
        input_length = len(input_task)
        
        # Extract ONLY the output portion (tokens generated after the input)
        output_only = generated_sample[input_length:] if len(generated_sample) > input_length else []
        
        # Compute 64-bit hash of input for fast lookup
        input_tensor = torch.tensor(input_task, dtype=torch.long)
        input_bytes = input_tensor.numpy().tobytes()
        input_hash = int(hashlib.sha256(input_bytes).hexdigest()[:16], 16)  # First 64 bits
        
        # Store tensors and hash
        self.generated_sequences.append(torch.tensor(output_only, dtype=torch.long))
        self.input_sequences.append(input_tensor)
        self.input_hashes.append(input_hash)
        self.finished.append(finished)
        self.match_info.append({
            'index': sample_index,
            'generated_length': generated_length,
            'matches': matches
        })
    
    def save(self, metadata: Dict[str, Any]) -> Path:
        """
        Save all collected samples to file
        
        Args:
            metadata: Metadata dictionary about the run
            
        Returns:
            Path to saved file
        """
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create output dictionary
        output_data = {
            'generated': self.generated_sequences,  # ONLY generated output tokens (after input)
            'inputs': self.input_sequences,         # Input sequences
            'input_hashes': self.input_hashes,      # 64-bit hashes for fast input lookup
            'finished': self.finished,              # Whether each sequence finished
            'match_info': self.match_info,          # Additional match information
            'metadata': metadata                     # Metadata about the run
        }
        
        torch.save(output_data, output_path)
        
        logging.info('\nSaved output to: %s', output_path)
        logging.info('  - Generated sequences: %d', len(self.generated_sequences))
        if output_path.exists():
            logging.info('  - File size: %.2f MB', output_path.stat().st_size / (1024 * 1024))
        
        return output_path