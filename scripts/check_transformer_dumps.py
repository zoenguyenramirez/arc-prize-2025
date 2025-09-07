#!/usr/bin/env python3
"""
Check transformer input/output dumps in temp/ directory
"""
import torch
import os
from pathlib import Path
import glob

def check_dumps():
    dump_dir = './temp'
    pattern = os.path.join(dump_dir, '*_input_ids.pt')
    dump_files = sorted(glob.glob(pattern))
    
    if not dump_files:
        print(f"No dump files found in {dump_dir}")
        return
    
    print(f"Found {len(dump_files)} dump files:")
    print("-" * 50)
    
    for dump_file in dump_files[-5:]:  # Show last 5 dumps
        print(f"\nFile: {os.path.basename(dump_file)}")
        
        data = torch.load(dump_file, map_location='cpu', weights_only=False)
        input_ids = data['input_ids']
        indices = data.get('indices', 'Not provided')
        
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Device: {input_ids.device}")
        print(f"  Dtype: {input_ids.dtype}")
        print(f"  Indices: {indices}")
        
        # Show some statistics
        if input_ids.numel() > 0:
            flat_ids = input_ids.flatten()
            print(f"  Token range: [{flat_ids.min().item()}, {flat_ids.max().item()}]")
            print(f"  Unique tokens: {len(torch.unique(flat_ids))}")
            
            # Show first few tokens from first sample
            if len(input_ids.shape) >= 2:
                first_sample = input_ids[0, :20] if input_ids.shape[1] > 20 else input_ids[0]
                print(f"  First 20 tokens: {first_sample.tolist()}")

def monitor_latest():
    """Monitor the most recent dump"""
    dump_dir = './temp'
    pattern = os.path.join(dump_dir, '*_input_ids.pt')
    dump_files = sorted(glob.glob(pattern))
    
    if dump_files:
        latest = dump_files[-1]
        print(f"\nLatest dump: {os.path.basename(latest)}")
        
        # Get file size and modification time
        stat = os.stat(latest)
        print(f"  Size: {stat.st_size / 1024:.1f} KB")
        print(f"  Modified: {Path(latest).stat().st_mtime}")
        
        # Next dump expected at
        # Extract epoch from filename
        basename = os.path.basename(latest)
        if basename.startswith('0000_'):
            print(f"  Next dump expected: epoch 100")
        elif basename.startswith('0100_'):
            print(f"  Next dump expected: epoch 200")

if __name__ == "__main__":
    print("=" * 50)
    print("Transformer Input/Output Dump Check")
    print("=" * 50)
    
    check_dumps()
    monitor_latest()
    
    print("\n" + "=" * 50)
    print("Note: Dumps occur every 100 epochs on the first batch")
    print("Files are saved to ./temp/*_input_ids.pt")
    print("=" * 50)