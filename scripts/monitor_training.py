#!/usr/bin/env python3
"""
Continuous monitoring script for pseudo-RL training
Verifies design expectations and reports violations
"""

import os
import json
import torch
import time
from datetime import datetime
from pathlib import Path
import glob

YELLOW = '\033[93m'
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_attempt_length_growth(scheduler_state, config):
    """Verify attempt_length follows expected exponential growth"""
    samples = scheduler_state['total_samples']
    samples_per_double = config['samples_per_double']
    expected_attempt = 0
    periods = samples // samples_per_double
    
    if samples >= samples_per_double:
        if periods == 1:
            expected_attempt = 1
        elif periods == 2:
            expected_attempt = 2
        elif periods == 3:
            expected_attempt = 4
        elif periods == 4:
            expected_attempt = 8
        elif periods == 5:
            expected_attempt = 16
        elif periods == 6:
            expected_attempt = 32
        elif periods == 7:
            expected_attempt = 64
        elif periods == 8:
            expected_attempt = 128
        elif periods == 9:
            expected_attempt = 256
        elif periods == 10:
            expected_attempt = 512
        elif periods >= 11:
            expected_attempt = 1000
    
    actual_attempt = scheduler_state.get('attempt_length', 0)
    
    return {
        'samples': samples,
        'expected': expected_attempt,
        'actual': actual_attempt,
        'correct': expected_attempt == actual_attempt,
        'next_growth_at': ((periods + 1) * samples_per_double) if periods < 11 else None
    }

def check_transformer_dumps():
    """Check transformer input dumps and analyze token patterns"""
    dump_dir = './temp'
    pattern = os.path.join(dump_dir, '*_input_ids.pt')
    dump_files = sorted(glob.glob(pattern))
    
    results = []
    for dump_file in dump_files[-3:]:  # Last 3 dumps
        try:
            data = torch.load(dump_file, map_location='cpu', weights_only=False)
            input_ids = data['input_ids']
            indices = data.get('indices', 'N/A')
            
            results.append({
                'file': os.path.basename(dump_file),
                'shape': input_ids.shape,
                'indices': indices,
                'token_range': [input_ids.min().item(), input_ids.max().item()],
                'unique_tokens': len(torch.unique(input_ids.flatten()))
            })
        except Exception as e:
            results.append({
                'file': os.path.basename(dump_file),
                'error': str(e)
            })
    
    return results

def monitor_iteration(run_dir):
    """Monitor current iteration progress"""
    # Find latest iteration
    iter_dirs = sorted([d for d in os.listdir(run_dir) if d.startswith('iter')])
    
    if not iter_dirs:
        return {'status': 'No iterations found'}
    
    latest_iter = iter_dirs[-1]
    iter_path = os.path.join(run_dir, latest_iter)
    
    # Read scheduler state
    scheduler_state_path = os.path.join(iter_path, 'scheduler_state.json')
    if os.path.exists(scheduler_state_path):
        with open(scheduler_state_path, 'r') as f:
            scheduler_state = json.load(f)
    else:
        scheduler_state = {'status': 'No scheduler state found'}
    
    # Check for checkpoints
    checkpoint_dirs = [d for d in os.listdir(iter_path) if os.path.isdir(os.path.join(iter_path, d))]
    latest_checkpoint = None
    
    for ckpt_dir in checkpoint_dirs:
        ckpt_path = os.path.join(iter_path, ckpt_dir, 'Transformer_latest.pt')
        if os.path.exists(ckpt_path):
            latest_checkpoint = ckpt_path
            break
    
    return {
        'iteration': latest_iter,
        'scheduler_state': scheduler_state,
        'checkpoint': latest_checkpoint
    }

def check_orchestration_state(run_dir):
    """Check orchestration states for violations"""
    states_dir = os.path.join(run_dir, '_orchestration_states')
    if not os.path.exists(states_dir):
        return {'status': 'No orchestration states found'}
    
    state_files = sorted(glob.glob(os.path.join(states_dir, 'state_*.json')))
    
    if not state_files:
        return {'status': 'No state files found'}
    
    # Read latest state
    with open(state_files[-1], 'r') as f:
        latest_state = json.load(f)
    
    return {
        'current_iteration': latest_state['current_iteration'],
        'total_iterations': latest_state['total_iterations'],
        'phase': latest_state['phase'],
        'state_files_count': len(state_files),
        'immutability_check': 'PASS' if len(state_files) == latest_state['current_iteration'] + 1 else 'FAIL'
    }

def main():
    run_dir = 'runs/pseudo_rl_20250904_085717'
    
    # Read config
    config_path = os.path.join(run_dir, '_config/scheduler_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n{BOLD}=== TRAINING MONITOR REPORT ==={RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run: {run_dir}")
    
    # 1. Check iteration status
    iter_status = monitor_iteration(run_dir)
    print(f"\n{BOLD}ITERATION STATUS:{RESET}")
    print(f"  Current: {iter_status['iteration']}")
    if 'scheduler_state' in iter_status and 'current_epoch' in iter_status['scheduler_state']:
        ss = iter_status['scheduler_state']
        print(f"  Epoch: {ss['current_epoch']}")
        print(f"  Samples: {ss['total_samples']:,}")
        print(f"  Attempt Length: {ss.get('attempt_length', 0)}")
    
    # 2. Check attempt length growth
    if 'scheduler_state' in iter_status and 'total_samples' in iter_status['scheduler_state']:
        growth_check = check_attempt_length_growth(iter_status['scheduler_state'], config)
        print(f"\n{BOLD}ATTEMPT LENGTH VERIFICATION:{RESET}")
        print(f"  Samples Processed: {growth_check['samples']:,}")
        print(f"  Expected Length: {growth_check['expected']}")
        print(f"  Actual Length: {growth_check['actual']}")
        
        if growth_check['correct']:
            print(f"  Status: {GREEN}✓ CORRECT{RESET}")
        else:
            print(f"  Status: {RED}✗ VIOLATION - Expected {growth_check['expected']}, got {growth_check['actual']}{RESET}")
        
        if growth_check['next_growth_at']:
            samples_until_growth = growth_check['next_growth_at'] - growth_check['samples']
            print(f"  Next Growth: at {growth_check['next_growth_at']:,} samples ({samples_until_growth:,} to go)")
    
    # 3. Check transformer dumps
    dumps = check_transformer_dumps()
    if dumps:
        print(f"\n{BOLD}TRANSFORMER DUMPS:{RESET}")
        for dump in dumps[-2:]:  # Show last 2
            if 'error' in dump:
                print(f"  {dump['file']}: {RED}Error - {dump['error']}{RESET}")
            else:
                print(f"  {dump['file']}:")
                print(f"    Shape: {dump['shape']}")
                print(f"    Token Range: {dump['token_range']}")
                print(f"    Unique Tokens: {dump['unique_tokens']}")
    
    # 4. Check orchestration state
    orch_state = check_orchestration_state(run_dir)
    print(f"\n{BOLD}ORCHESTRATION STATE:{RESET}")
    print(f"  Iteration: {orch_state.get('current_iteration', 'N/A')} / {orch_state.get('total_iterations', 'N/A')}")
    print(f"  Phase: {orch_state.get('phase', 'N/A')}")
    print(f"  Immutability Check: ", end='')
    if orch_state.get('immutability_check') == 'PASS':
        print(f"{GREEN}✓ PASS{RESET}")
    elif orch_state.get('immutability_check') == 'FAIL':
        print(f"{RED}✗ FAIL - State files being overwritten!{RESET}")
    else:
        print(f"{YELLOW}? UNKNOWN{RESET}")
    
    # 5. Check for errors
    print(f"\n{BOLD}DESIGN VIOLATIONS:{RESET}")
    violations = []
    
    # Check attempt length
    if 'scheduler_state' in iter_status and 'total_samples' in iter_status['scheduler_state']:
        growth_check = check_attempt_length_growth(iter_status['scheduler_state'], config)
        if not growth_check['correct']:
            violations.append(f"Attempt length mismatch: expected {growth_check['expected']}, got {growth_check['actual']}")
    
    # Check immutability
    if orch_state.get('immutability_check') == 'FAIL':
        violations.append("Orchestration states being overwritten - violates immutability design")
    
    if violations:
        for v in violations:
            print(f"  {RED}✗ {v}{RESET}")
    else:
        print(f"  {GREEN}✓ No violations detected{RESET}")
    
    print(f"\n{'=' * 60}")

if __name__ == "__main__":
    main()