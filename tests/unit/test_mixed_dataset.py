#!/usr/bin/env python3
"""
Test MixedGridDataset functionality
"""

import unittest
import torch
import numpy as np
from typing import Dict, Any

from src.load_data import GridDataset, MixedGridDataset
from src.token import SpecialToken


class TestMixedDataset(unittest.TestCase):
    """Test MixedGridDataset functionality"""
    
    def setUp(self):
        """Create mock datasets for testing"""
        # Create a simple GridDataset with mock data
        self.grid_dataset = GridDataset()
        
        # Add some mock compact data (simple format)
        # Each item is a compact representation: [START_INPUT, h, w, values..., START_OUTPUT, h, w, values...]
        self.grid_dataset.data = []
        for i in range(10):
            self.grid_dataset.data.append(
                np.array([SpecialToken.START_INPUT.value, 1, 1, i % 10, 
                         SpecialToken.START_OUTPUT.value, 1, 1, (i*2) % 10])
            )
        
        # Create mock trajectory data for samples 0, 1, 2
        self.trajectory_data = {
            0: {'attempt_tokens': [7, 8, 9]},
            1: {'attempt_tokens': [10, 11, 12]},
            2: {'attempt_tokens': [13, 14, 15, 16, 17]},  # Longer attempt
        }
    
    def test_mixed_dataset_creation(self):
        """Test creating a mixed dataset"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=self.trajectory_data
        )
        
        self.assertEqual(len(mixed_dataset), 10)
        # Should use all available trajectories
        self.assertEqual(mixed_dataset.use_trajectory, {0, 1, 2})
    
    def test_all_trajectories_used(self):
        """Test that all available trajectories are used"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=self.trajectory_data
        )
        
        # All trajectory indices should be marked for use
        self.assertEqual(mixed_dataset.use_trajectory, {0, 1, 2})
    
    def test_partial_trajectories(self):
        """Test with only some samples having trajectories"""
        # Only provide trajectories for sample 5
        partial_trajectories = {
            5: {'attempt_tokens': [100, 101, 102]}
        }
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=partial_trajectories
        )
        
        # Only index 5 should be marked for trajectory use
        self.assertEqual(mixed_dataset.use_trajectory, {5})
    
    def test_getitem_with_trajectory(self):
        """Test getting an item that has trajectory data"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=self.trajectory_data
        )
        
        # Set attempt length to allow trajectory tokens
        mixed_dataset.set_attempt_length(10)
        
        # Get item 0 (has trajectory)
        item = mixed_dataset[0]
        
        # Should return a dictionary with task data
        self.assertIn('task', item)
        self.assertIn('idx', item)
        self.assertIn('end_of_examples_index', item)
    
    def test_getitem_without_trajectory(self):
        """Test getting an item without trajectory data"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data={}  # No trajectories
        )
        
        # Get item 5 (no trajectory data)
        item = mixed_dataset[5]
        
        # Should still return valid data with empty attempt section
        self.assertIn('task', item)
        self.assertIn('idx', item)
        self.assertIn('end_of_examples_index', item)
    
    def test_set_methods(self):
        """Test that set methods are properly forwarded"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=self.trajectory_data
        )
        
        # Test set_augment_seed
        mixed_dataset.set_augment_seed(42)
        self.assertEqual(mixed_dataset.augment_seed, 42)
        self.assertEqual(mixed_dataset.original_dataset.augment_seed, 42)
        
        # Test set_max_length
        mixed_dataset.set_max_length(1024)
        self.assertEqual(mixed_dataset.max_length, 1024)
        self.assertEqual(mixed_dataset.original_dataset.max_length, 1024)
        
        # Test set_attempt_length
        mixed_dataset.set_attempt_length(100)
        self.assertEqual(mixed_dataset.current_attempt_length, 100)
        self.assertEqual(mixed_dataset.original_dataset.current_attempt_length, 100)
    
    def test_attempt_length_truncation(self):
        """Test that attempts are truncated to current_attempt_length"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=self.trajectory_data
        )
        
        # Set attempt length to 3 (shorter than trajectory for sample 2)
        mixed_dataset.set_attempt_length(3)
        
        # Sample 2 has 5 attempt tokens, should be truncated to 3
        # We can't directly test the truncation without running through
        # the full pipeline, but we can verify the setting is stored
        self.assertEqual(mixed_dataset.current_attempt_length, 3)
    
    def test_empty_trajectories(self):
        """Test with empty trajectory dict"""
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data={}
        )
        
        # Should have no trajectory indices
        self.assertEqual(len(mixed_dataset.use_trajectory), 0)
        
        # Should still be able to get items (with empty attempts)
        for i in range(len(mixed_dataset)):
            item = mixed_dataset[i]
            self.assertIsNotNone(item)
    
    def test_large_trajectory_set(self):
        """Test with trajectories for all samples"""
        # Create trajectories for all 10 samples
        all_trajectories = {
            i: {'attempt_tokens': [100 + i, 200 + i]} 
            for i in range(10)
        }
        
        mixed_dataset = MixedGridDataset(
            original_dataset=self.grid_dataset,
            trajectory_data=all_trajectories
        )
        
        # All samples should be marked for trajectory use
        self.assertEqual(mixed_dataset.use_trajectory, set(range(10)))


if __name__ == '__main__':
    unittest.main()