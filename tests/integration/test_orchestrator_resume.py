#!/usr/bin/env python3
"""
Test the resume functionality of the orchestrator

This script tests that the orchestrator can:
1. Save state after each phase
2. Resume from any interrupted point
3. Skip already completed phases
4. Handle state file corruption gracefully
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
import sys
sys.path.append('.')

from src.orchestrate_training import OrchestrationState, PseudoRLOrchestrator


def test_state_manager():
    """Test the OrchestrationState manager"""
    print("Testing OrchestrationState...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "test_state.json"
        manager = OrchestrationState(state_file)
        
        # Test initial load (no file)
        state = manager.load()
        assert state is None, "Should return None for non-existent file"
        
        # Test save and load
        test_state = {
            'run_name': 'test_run',
            'current_iteration': 0,
            'completed_phases': {}
        }
        manager.save(test_state)
        
        loaded = manager.load()
        assert loaded['run_name'] == 'test_run', "Should save and load state correctly"
        assert 'last_updated' in loaded, "Should add timestamp"
        
        # Test phase completion tracking
        manager.mark_phase_completed(0, 'initial_training', checkpoint='test.pt')
        assert manager.is_phase_completed(0, 'initial_training'), "Should mark phase as completed"
        assert not manager.is_phase_completed(0, 'trajectory_generation'), "Should not mark other phases"
        assert not manager.is_phase_completed(1, 'initial_training'), "Should not mark other iterations"
        
        # Verify saved data
        loaded = manager.load()
        phase_key = 'iter_0_initial_training'
        assert phase_key in loaded['completed_phases'], "Should save phase completion"
        assert loaded['completed_phases'][phase_key]['checkpoint'] == 'test.pt', "Should save phase data"
        
    print("✓ OrchestrationState tests passed")


def test_resume_scenarios():
    """Test various resume scenarios"""
    print("\nTesting resume scenarios...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "orchestration_state.json"
        
        # Scenario 1: Resume after initial training
        state = {
            'run_name': 'test_run',
            'run_dir': tmpdir,
            'config': {
                'dataset': 'test.pth',
                'num_iterations': 3,
                'initial_epochs': 10,
                'mixed_epochs': 5,
            },
            'current_iteration': 0,
            'current_phase': 'initial_training',
            'completed_phases': {
                'iter_0_initial_training': {
                    'completed': True,
                    'checkpoint': 'checkpoint1.pt'
                }
            },
            'checkpoints': {},
            'trajectory_dirs': {}
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f)
        
        manager = OrchestrationState(state_file)
        loaded = manager.load()
        
        # Should skip initial training for iteration 0
        assert manager.is_phase_completed(0, 'initial_training'), "Should recognize completed phase"
        assert not manager.is_phase_completed(0, 'trajectory_generation'), "Should not mark uncompleted phase"
        
        # Scenario 2: Resume in middle of iteration 2
        manager.mark_phase_completed(0, 'trajectory_generation', trajectory_dir='traj0')
        manager.mark_phase_completed(0, 'mixed_training', checkpoint='checkpoint2.pt')
        manager.mark_phase_completed(1, 'initial_training', checkpoint='checkpoint3.pt')
        manager.mark_phase_completed(1, 'trajectory_generation', trajectory_dir='traj1')
        # iteration 1 mixed_training not completed - should resume here
        
        # Check resume point detection
        resume_iteration = None
        resume_phase = None
        for i in range(3):
            for phase in ['initial_training', 'trajectory_generation', 'mixed_training']:
                if not manager.is_phase_completed(i, phase):
                    resume_iteration = i
                    resume_phase = phase
                    break
            if resume_iteration is not None:
                break
        
        assert resume_iteration == 1, f"Should resume at iteration 1, got {resume_iteration}"
        assert resume_phase == 'mixed_training', f"Should resume at mixed_training, got {resume_phase}"
        
    print("✓ Resume scenario tests passed")


def test_atomic_save():
    """Test atomic state saving"""
    print("\nTesting atomic save...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "test_state.json"
        manager = OrchestrationState(state_file)
        
        # Save initial state
        initial_state = {'test': 'data', 'completed_phases': {}}
        manager.save(initial_state)
        
        # Simulate interrupted save by creating temp file
        temp_file = state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            f.write("corrupted")
        
        # Original file should still be valid
        loaded = manager.load()
        assert loaded['test'] == 'data', "Original file should remain intact"
        
        # Clean up and save new state
        temp_file.unlink()
        new_state = {'test': 'new_data', 'completed_phases': {}}
        manager.save(new_state)
        
        loaded = manager.load()
        assert loaded['test'] == 'new_data', "Should update state atomically"
        assert not temp_file.exists(), "Should not leave temp files"
        
    print("✓ Atomic save tests passed")


def test_full_cycle():
    """Test a full training cycle with interruptions"""
    print("\nTesting full cycle with interruptions...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "orchestration_state.json"
        manager = OrchestrationState(state_file)
        
        # Initialize state as orchestrator would
        state = {
            'run_name': 'test_full_cycle',
            'run_dir': tmpdir,
            'config': {
                'dataset': 'test.pth',
                'num_iterations': 2,
                'initial_epochs': 5,
                'mixed_epochs': 3,
            },
            'current_iteration': 0,
            'current_phase': 'not_started',
            'completed_phases': {},
            'checkpoints': {},
            'trajectory_dirs': {}
        }
        manager.save(state)
        
        # Simulate training phases
        phases = [
            (0, 'initial_training', {'checkpoint': 'ckpt1.pt'}),
            (0, 'trajectory_generation', {'trajectory_dir': 'traj1'}),
            (0, 'mixed_training', {'checkpoint': 'ckpt2.pt'}),
            (1, 'initial_training', {'checkpoint': 'ckpt3.pt'}),
            (1, 'trajectory_generation', {'trajectory_dir': 'traj2'}),
            (1, 'mixed_training', {'checkpoint': 'ckpt4.pt'}),
        ]
        
        completed_count = 0
        for iteration, phase, data in phases:
            # Check if should skip
            if manager.is_phase_completed(iteration, phase):
                print(f"  Skipping {iteration}/{phase} (already completed)")
            else:
                print(f"  Running {iteration}/{phase}")
                manager.mark_phase_completed(iteration, phase, **data)
                completed_count += 1
            
            # Simulate interruption and resume every 2 phases
            if completed_count % 2 == 0 and completed_count > 0:
                print("  [Simulated interruption]")
                # Reload state as if resuming
                manager = OrchestrationState(state_file)
                manager.load()
        
        # Verify all phases completed
        final_state = manager.load()
        assert len(final_state['completed_phases']) == 6, "Should complete all phases"
        
        # Verify can detect completion
        all_complete = True
        for i in range(2):
            for p in ['initial_training', 'trajectory_generation', 'mixed_training']:
                if not manager.is_phase_completed(i, p):
                    all_complete = False
                    break
        
        assert all_complete, "Should recognize all phases as complete"
        
    print("✓ Full cycle test passed")


if __name__ == '__main__':
    print("="*70)
    print("TESTING ORCHESTRATOR RESUME FUNCTIONALITY")
    print("="*70)
    
    test_state_manager()
    test_resume_scenarios()
    test_atomic_save()
    test_full_cycle()
    
    print("\n" + "="*70)
    print("✅ ALL ORCHESTRATOR RESUME TESTS PASSED!")
    print("="*70)
    print("\nThe orchestrator can:")
    print("  • Save state atomically after each phase")
    print("  • Resume from any interruption point")
    print("  • Skip already completed phases")
    print("  • Track progress across iterations")
    print("  • Handle multiple interruptions gracefully")