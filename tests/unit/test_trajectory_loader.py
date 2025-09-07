#!/usr/bin/env python3
"""
Test trajectory loader functionality
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path

from src.trajectory_loader import TrajectoryLoader
from src.token import SpecialToken


class TestTrajectoryLoader(unittest.TestCase):
    """Test trajectory loader functionality"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.trajectory_folder = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_empty_folder(self):
        """Test loading from empty folder"""
        loader = TrajectoryLoader(self.trajectory_folder)
        trajectories = loader.load_trajectories()
        
        self.assertEqual(len(trajectories), 0)
    
    def test_load_trajectory_file(self):
        """Test loading a trajectory file"""
        # Create mock trajectory file
        trajectory_file = self.trajectory_folder / "trajectories_20250101_120000.pt"
        
        mock_data = {
            'generated': [
                torch.tensor([1, 2, 3]),  # Generated output for sample 0
                torch.tensor([4, 5, 6]),  # Generated output for sample 1
            ],
            'inputs': [
                torch.tensor([7, 8, 9]),   # Input for sample 0
                torch.tensor([10, 11, 12]), # Input for sample 1
            ],
            'input_hashes': [12345, 67890],
            'finished': [True, False],
            'match_info': [
                {'index': 0, 'generated_length': 3, 'matches': True},
                {'index': 1, 'generated_length': 3, 'matches': False},
            ],
            'metadata': {
                'timestamp': '20250101_120000',
                'total_sequences': 2,
                'max_tokens': 1000,
                'batch_size': 1,
                'temperature': 0.0,
                'device': 'cuda',
                'trajectories': [
                    torch.tensor([7, 8, 9, 1, 2, 3]),   # Full trajectory 0
                    torch.tensor([10, 11, 12, 4, 5, 6]), # Full trajectory 1
                ],
                'ground_truth_outputs': [
                    [13, 14, 15],  # Ground truth for sample 0
                    [16, 17, 18],  # Ground truth for sample 1
                ],
            }
        }
        
        torch.save(mock_data, trajectory_file)
        
        # Load trajectories
        loader = TrajectoryLoader(self.trajectory_folder)
        trajectories = loader.load_trajectories()
        
        # Check loaded data
        self.assertEqual(len(trajectories), 2)
        
        # Check sample 0
        self.assertIn(0, trajectories)
        traj0 = trajectories[0]
        self.assertEqual(traj0['attempt_tokens'], [1, 2, 3])
        self.assertEqual(traj0['ground_truth'], [13, 14, 15])
        
        # Check sample 1
        self.assertIn(1, trajectories)
        traj1 = trajectories[1]
        self.assertEqual(traj1['attempt_tokens'], [4, 5, 6])
        self.assertEqual(traj1['ground_truth'], [16, 17, 18])
    
    def test_format_for_training(self):
        """Test formatting trajectory for training"""
        # Create mock trajectory file
        trajectory_file = self.trajectory_folder / "trajectories_test.pt"
        
        mock_data = {
            'generated': [torch.tensor([1, 2, 3])],
            'inputs': [torch.tensor([4, 5, 6])],
            'input_hashes': [12345],
            'finished': [True],
            'match_info': [{'index': 0, 'generated_length': 3, 'matches': True}],
            'metadata': {
                'ground_truth_outputs': [[7, 8, 9]],
            }
        }
        
        torch.save(mock_data, trajectory_file)
        
        # Load and format
        loader = TrajectoryLoader(self.trajectory_folder)
        loader.load_trajectories()
        
        formatted = loader.format_for_training(0)
        
        self.assertIsNotNone(formatted)
        self.assertEqual(formatted['attempt_tokens'], [1, 2, 3])
        self.assertEqual(formatted['ground_truth'], [7, 8, 9])
        
        # Try non-existent sample
        formatted_none = loader.format_for_training(999)
        self.assertIsNone(formatted_none)
    
    def test_merge_with_dataset(self):
        """Test merging trajectory data with dataset"""
        # Create mock trajectory file
        trajectory_file = self.trajectory_folder / "trajectories_test.pt"
        
        mock_data = {
            'generated': [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
                torch.tensor([7, 8, 9]),
            ],
            'inputs': [
                torch.tensor([10, 11, 12]),
                torch.tensor([13, 14, 15]),
                torch.tensor([16, 17, 18]),
            ],
            'input_hashes': [1, 2, 3],
            'finished': [True, True, False],
            'match_info': [
                {'index': 0, 'generated_length': 3, 'matches': True},
                {'index': 1, 'generated_length': 3, 'matches': True},
                {'index': 2, 'generated_length': 3, 'matches': False},
            ],
            'metadata': {
                'ground_truth_outputs': [
                    [19, 20, 21],
                    [22, 23, 24],
                    [25, 26, 27],
                ],
            }
        }
        
        torch.save(mock_data, trajectory_file)
        
        # Load trajectories
        loader = TrajectoryLoader(self.trajectory_folder)
        loader.load_trajectories()
        
        # Merge with dataset of size 2 (should only include first 2)
        merged = loader.merge_with_dataset(dataset_size=2)
        
        self.assertEqual(len(merged), 2)
        self.assertIn(0, merged)
        self.assertIn(1, merged)
        self.assertNotIn(2, merged)  # Index 2 is beyond dataset size
        
        # Check merged data
        self.assertEqual(merged[0]['attempt_tokens'], [1, 2, 3])
        self.assertEqual(merged[1]['attempt_tokens'], [4, 5, 6])


if __name__ == '__main__':
    unittest.main()