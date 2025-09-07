#!/usr/bin/env python3
"""
Test data preparation with attempt sections
"""

import unittest
import torch
from src.load_data import GridDataset
from src.token import SpecialToken
from src.utils.grid_data_process import create_special_token_sequence


class TestDataPreparation(unittest.TestCase):
    
    def setUp(self):
        """Set up test dataset"""
        self.dataset = GridDataset()
        # Add properly formatted test data with all required tokens
        # Format: [START_INPUT, height, width, grid_data..., START_OUTPUT, height, width, grid_data..., ...]
        self.dataset.data = [
            # A simple task with one example and one test case
            [
                13,  # START_INPUT (example input)
                1, 2,  # Grid dimensions (1x2)
                0, 1,  # Grid data
                15,  # START_OUTPUT (example output)
                1, 2,  # Grid dimensions
                1, 0,  # Grid data (flipped)
                13,  # START_INPUT (test input)
                1, 2,  # Grid dimensions
                2, 3,  # Grid data
                15,  # START_OUTPUT (test output)
                1, 2,  # Grid dimensions  
                3, 2,  # Grid data (flipped)
            ],
        ]
    
    def test_empty_attempt_section(self):
        """Test that empty attempt section is added for original data"""
        self.dataset.set_attempt_length(0)
        
        # Get a sample
        sample = self.dataset[0]
        task = sample['task']
        
        # Check that ATTEMPT_START and ATTEMPT_END tokens are present
        attempt_start_found = False
        attempt_end_found = False
        
        for token in task:
            if isinstance(token, list) and len(token) >= 5:
                if token[0] == SpecialToken.ATTEMPT_START.value:
                    attempt_start_found = True
                elif token[0] == SpecialToken.ATTEMPT_END.value:
                    attempt_end_found = True
        
        self.assertTrue(attempt_start_found, "ATTEMPT_START token not found")
        self.assertTrue(attempt_end_found, "ATTEMPT_END token not found")
    
    def test_attempt_length_setting(self):
        """Test setting attempt length"""
        self.dataset.set_attempt_length(100)
        self.assertEqual(self.dataset.current_attempt_length, 100)
        
        self.dataset.set_attempt_length(500)
        self.assertEqual(self.dataset.current_attempt_length, 500)
    
    def test_trajectory_data_truncation(self):
        """Test that trajectory data is truncated to current attempt length"""
        # Set up trajectory data with attempt tokens
        self.dataset.trajectory_data = {
            0: {
                'attempt_tokens': [[1, 0, 0, 0, 0]] * 100  # 100 attempt tokens
            }
        }
        
        # Set attempt length to 50
        self.dataset.set_attempt_length(50)
        
        # Get sample
        sample = self.dataset[0]
        task = sample['task']
        
        # Count tokens between ATTEMPT_START and ATTEMPT_END
        in_attempt = False
        attempt_tokens = 0
        
        for token in task:
            if isinstance(token, list) and len(token) >= 5:
                if token[0] == SpecialToken.ATTEMPT_START.value:
                    in_attempt = True
                elif token[0] == SpecialToken.ATTEMPT_END.value:
                    in_attempt = False
                elif in_attempt:
                    attempt_tokens += 1
        
        # Should be truncated to 50
        self.assertEqual(attempt_tokens, 50, f"Expected 50 attempt tokens, got {attempt_tokens}")
    
    def test_no_pad_tokens(self):
        """Test that no PAD tokens are added in attempt sections"""
        self.dataset.set_attempt_length(10)
        
        # Get sample
        sample = self.dataset[0]
        task = sample['task']
        
        # Check no PAD tokens in attempt section
        in_attempt = False
        for token in task:
            if isinstance(token, list) and len(token) >= 5:
                if token[0] == SpecialToken.ATTEMPT_START.value:
                    in_attempt = True
                elif token[0] == SpecialToken.ATTEMPT_END.value:
                    in_attempt = False
                elif in_attempt:
                    self.assertNotEqual(token[0], SpecialToken.PAD.value, 
                                       "PAD token found in attempt section")


if __name__ == '__main__':
    unittest.main()