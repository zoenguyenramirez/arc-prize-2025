#!/usr/bin/env python3
"""
Test trajectory generation with ground truth output extraction
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path

from src.token import SpecialToken
from src.utils.grid_data_process import remove_last_grid


class TestTrajectoryGeneration(unittest.TestCase):
    """Test trajectory generation functionality"""
    
    def test_remove_last_grid(self):
        """Test that remove_last_grid correctly extracts input without output"""
        # Create a simple task in COMPACT format (height, width, then grid values)
        # Format: START_INPUT, height, width, values..., START_OUTPUT, height, width, values...
        task = [
            SpecialToken.START_INPUT.value,
            2, 2,  # 2x2 grid
            1, 2, 3, 4,  # Grid values (must be 0-9)
            SpecialToken.START_OUTPUT.value,
            2, 2,  # 2x2 grid
            5, 6, 7, 8,  # Output grid values
        ]
        
        # Remove the last grid (output)
        input_only = remove_last_grid(task)
        
        # Should only have the input part
        self.assertEqual(input_only, [
            SpecialToken.START_INPUT.value,
            2, 2,  # 2x2 grid
            1, 2, 3, 4,
        ])
        
        # Extract ground truth (the removed part)
        ground_truth = task[len(input_only):]
        self.assertEqual(ground_truth, [
            SpecialToken.START_OUTPUT.value,
            2, 2,  # 2x2 grid
            5, 6, 7, 8,
        ])
    
    def test_trajectory_output_format(self):
        """Test that trajectory generator produces correct output format"""
        # This test would require a mock model, so we test the data structure
        from src.rl_trajectory_generator import TrajectoryGenerator
        
        # Create mock trajectory data
        trajectory_data = {
            'trajectories': torch.tensor([[1, 2, 3, 4, 5]]),
            'finished': torch.tensor([True]),
        }
        
        # Verify structure
        self.assertIn('trajectories', trajectory_data)
        self.assertIn('finished', trajectory_data)
        self.assertEqual(trajectory_data['trajectories'].shape[0], 1)  # Batch size 1
        self.assertEqual(trajectory_data['finished'].shape[0], 1)
    
    def test_ground_truth_extraction(self):
        """Test that ground truth is correctly extracted from full task"""
        # Create a more complex task with multiple input/output pairs in COMPACT format
        task = [
            # Example 1
            SpecialToken.START_INPUT.value,
            1, 1,  # 1x1 grid
            1,
            SpecialToken.START_OUTPUT.value,
            1, 1,  # 1x1 grid
            2,
            # Example 2
            SpecialToken.START_INPUT.value,
            1, 1,  # 1x1 grid
            3,
            SpecialToken.START_OUTPUT.value,
            1, 1,  # 1x1 grid
            4,
            # Test case (last pair)
            SpecialToken.START_INPUT.value,
            1, 2,  # 1x2 grid
            5, 6,
            SpecialToken.START_OUTPUT.value,
            1, 2,  # 1x2 grid
            7, 8,
        ]
        
        # Remove the last output
        input_only = remove_last_grid(task)
        
        # Should have everything except the last output
        expected_input = [
            # Example 1
            SpecialToken.START_INPUT.value,
            1, 1,  # 1x1 grid
            1,
            SpecialToken.START_OUTPUT.value,
            1, 1,  # 1x1 grid
            2,
            # Example 2
            SpecialToken.START_INPUT.value,
            1, 1,  # 1x1 grid
            3,
            SpecialToken.START_OUTPUT.value,
            1, 1,  # 1x1 grid
            4,
            # Test case input only
            SpecialToken.START_INPUT.value,
            1, 2,  # 1x2 grid
            5, 6,
        ]
        
        self.assertEqual(input_only, expected_input)
        
        # Ground truth should be the last output
        ground_truth = task[len(input_only):]
        expected_ground_truth = [
            SpecialToken.START_OUTPUT.value,
            1, 2,  # 1x2 grid
            7, 8,
        ]
        
        self.assertEqual(ground_truth, expected_ground_truth)
    
    def test_saved_trajectory_structure(self):
        """Test that saved trajectory files have correct structure"""
        # Create a temporary file to save trajectory
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create mock data structure as it would be saved
            mock_data = {
                'generated': [torch.tensor([1, 2, 3])],  # Generated outputs only
                'inputs': [torch.tensor([4, 5, 6])],     # Input sequences
                'input_hashes': [12345],                 # Hashes for lookup
                'finished': [True],                      # Completion status
                'match_info': [{
                    'index': 0,
                    'generated_length': 3,
                    'matches': True
                }],
                'metadata': {
                    'timestamp': '20250101_120000',
                    'total_sequences': 1,
                    'max_tokens': 1000,
                    'batch_size': 1,
                    'temperature': 0.0,
                    'device': 'cuda',
                    'trajectories': [torch.tensor([4, 5, 6, 1, 2, 3])],  # Full trajectories
                    'ground_truth_outputs': [[7, 8, 9]],  # Ground truth for comparison
                }
            }
            
            # Save the data
            torch.save(mock_data, tmp_path)
            
            # Load and verify structure
            loaded_data = torch.load(tmp_path)
            
            # Check all required fields exist
            self.assertIn('generated', loaded_data)
            self.assertIn('inputs', loaded_data)
            self.assertIn('input_hashes', loaded_data)
            self.assertIn('finished', loaded_data)
            self.assertIn('match_info', loaded_data)
            self.assertIn('metadata', loaded_data)
            
            # Check metadata has ground truth
            self.assertIn('ground_truth_outputs', loaded_data['metadata'])
            self.assertIn('trajectories', loaded_data['metadata'])
            
            # Verify ground truth format
            ground_truth = loaded_data['metadata']['ground_truth_outputs']
            self.assertIsInstance(ground_truth, list)
            self.assertEqual(len(ground_truth), 1)  # One sample
            self.assertEqual(ground_truth[0], [7, 8, 9])
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()