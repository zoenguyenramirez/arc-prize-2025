#!/usr/bin/env python3
"""
Unit tests for UnifiedTrainingScheduler
"""

import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schedulars import UnifiedTrainingScheduler


class TestUnifiedScheduler(unittest.TestCase):
    """Test suite for UnifiedTrainingScheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix='test_scheduler_')
        self.model = nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_scheduler_initialization(self):
        """Test scheduler can be initialized"""
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir,
            warmup_epochs=10,
            total_epochs=100,
            samples_per_double=10000
        )
        self.assertIsNotNone(scheduler)
        self.assertEqual(scheduler.samples_per_double, 10000)
        self.assertEqual(scheduler.max_attempt_tokens, 1000)
    
    def test_attempt_length_progression(self):
        """Test exponential growth of attempt length"""
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir,
            samples_per_double=10000
        )
        
        test_cases = [
            (0, 0),       # 0 samples -> 0 tokens
            (5000, 0),    # 5k samples -> 0 tokens
            (10000, 1),   # 10k samples -> 1 token
            (20000, 2),   # 20k samples -> 2 tokens
            (30000, 4),   # 30k samples -> 4 tokens
            (40000, 8),   # 40k samples -> 8 tokens
            (50000, 16),  # 50k samples -> 16 tokens
            (60000, 32),  # 60k samples -> 32 tokens
            (70000, 64),  # 70k samples -> 64 tokens
            (80000, 128), # 80k samples -> 128 tokens
            (90000, 256), # 90k samples -> 256 tokens
            (100000, 512), # 100k samples -> 512 tokens
            (110000, 1000), # 110k+ samples -> 1000 tokens (capped)
        ]
        
        for samples, expected_length in test_cases:
            scheduler.total_samples = samples
            actual_length = scheduler.get_current_attempt_length()
            self.assertEqual(actual_length, expected_length,
                           f"At {samples} samples: expected {expected_length}, got {actual_length}")
    
    def test_step_functionality(self):
        """Test that step() correctly updates counters"""
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir
        )
        
        initial_samples = scheduler.total_samples
        initial_steps = scheduler.total_steps
        
        # Step with default batch size
        scheduler.step()
        self.assertEqual(scheduler.total_samples, initial_samples + scheduler.batch_size)
        self.assertEqual(scheduler.total_steps, initial_steps + 1)
        
        # Step with custom batch size
        scheduler.step(batch_size=8)
        self.assertEqual(scheduler.total_samples, initial_samples + scheduler.batch_size + 8)
        self.assertEqual(scheduler.total_steps, initial_steps + 2)
    
    def test_save_and_load_state(self):
        """Test state persistence"""
        scheduler1 = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir,
            samples_per_double=10000
        )
        
        # Set some state
        scheduler1.current_epoch = 5
        scheduler1.total_samples = 25000
        scheduler1.total_steps = 6250
        scheduler1.save_state()
        
        # Create new scheduler and load state
        optimizer2 = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler2 = UnifiedTrainingScheduler(
            optimizer=optimizer2,
            run_dir=self.test_dir
        )
        
        # Verify state was loaded
        self.assertEqual(scheduler2.current_epoch, 5)
        self.assertEqual(scheduler2.total_samples, 25000)
        self.assertEqual(scheduler2.total_steps, 6250)
        self.assertEqual(scheduler2.get_current_attempt_length(), 2)  # At 25k samples
    
    def test_rollout_checking(self):
        """Test rollout frequency checking"""
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir
        )
        
        # Test at exact rollout point
        scheduler.total_samples = 50000
        self.assertTrue(scheduler.should_rollout(50000))
        
        # Test just after rollout point
        scheduler.total_samples = 50001
        self.assertFalse(scheduler.should_rollout(50000))
        
        # Test at zero (should not trigger)
        scheduler.total_samples = 0
        self.assertFalse(scheduler.should_rollout(50000))
        
        # Test with no rollout frequency
        scheduler.total_samples = 50000
        self.assertFalse(scheduler.should_rollout(None))
    
    def test_get_status(self):
        """Test status reporting"""
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir,
            samples_per_double=10000
        )
        
        scheduler.total_samples = 30000
        scheduler.current_epoch = 75
        
        status = scheduler.get_status()
        
        self.assertIn('epoch', status)
        self.assertIn('lr', status)
        self.assertIn('attempt_length', status)
        self.assertIn('total_samples', status)
        self.assertIn('total_steps', status)
        self.assertEqual(status['epoch'], 75)
        self.assertEqual(status['total_samples'], 30000)
        self.assertEqual(status['attempt_length'], 4)  # At 30k samples
    
    def test_dry_run(self):
        """Test dry-run functionality"""
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir,
            total_epochs=10,
            samples_per_double=10000
        )
        
        # Capture dry-run output
        result = scheduler.dry_run(
            dataset_size=400,
            batch_size=4,
            rollout_frequency_samples=2000
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['dataset_size'], 400)
        self.assertEqual(result['batch_size'], 4)
        self.assertEqual(result['total_epochs'], 10)
        self.assertIn('attempt_changes', result)
        self.assertIn('rollout_epochs', result)
    
    def test_dry_run_with_graphs(self):
        """Test dry-run with graph generation"""
        import os
        
        scheduler = UnifiedTrainingScheduler(
            optimizer=self.optimizer,
            run_dir=self.test_dir,
            total_epochs=50,
            samples_per_double=5000
        )
        
        graph_path = os.path.join(self.test_dir, 'test_preview.png')
        
        # Run dry-run with graphs
        result = scheduler.dry_run(
            dataset_size=400,
            batch_size=4,
            rollout_frequency_samples=10000,
            save_graphs=True,
            graph_path=graph_path
        )
        
        # Check graph was created
        self.assertTrue(os.path.exists(graph_path))
        self.assertGreater(os.path.getsize(graph_path), 0)


if __name__ == '__main__':
    unittest.main()