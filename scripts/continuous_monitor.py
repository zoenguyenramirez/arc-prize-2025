#!/usr/bin/env python3
"""
Continuous monitoring daemon for pseudo-RL training
Runs every 10 minutes and reports status/violations
"""

import time
import subprocess
from datetime import datetime

MONITOR_INTERVAL = 600  # 10 minutes

def run_monitor():
    """Run the monitoring script and capture output"""
    try:
        result = subprocess.run(['python', 'scripts/monitor_training.py'], 
                              capture_output=True, text=True, timeout=30)
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"ERROR: Monitor script timed out"
    except Exception as e:
        return f"ERROR: {e}"

print("=" * 80)
print("CONTINUOUS TRAINING MONITOR - Starting")
print(f"Will check every {MONITOR_INTERVAL} seconds (10 minutes)")
print("Press Ctrl+C to stop")
print("=" * 80)

iteration = 0
while True:
    iteration += 1
    print(f"\n{'#' * 80}")
    print(f"MONITOR CHECK #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('#' * 80)
    
    output = run_monitor()
    print(output)
    
    # Check for violations in output
    if "VIOLATION" in output or "FAIL" in output or "Error" in output:
        print("\n" + "!" * 80)
        print("!!! ATTENTION: VIOLATIONS OR ERRORS DETECTED !!!")
        print("!" * 80)
    
    print(f"\nNext check in {MONITOR_INTERVAL} seconds...")
    time.sleep(MONITOR_INTERVAL)