#!/usr/bin/env python3
"""
Professional comparison tool for inference outputs.
Compares two .pt files from either src.sample or trajectory generator.
"""

import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_inference_file(filepath: Path) -> Dict[str, Any]:
    """Load and validate an inference output file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = torch.load(filepath, map_location='cpu')
    
    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"Invalid file format: expected dict, got {type(data)}")
    
    return data


def extract_token_sequences(data: Dict[str, Any]) -> Tuple[List[torch.Tensor], str]:
    """
    Extract token sequences from either format.
    Returns (sequences, format_type)
    """
    # Both formats now store 'generated' with ONLY output tokens
    if 'generated' in data:
        # Both src.sample and trajectory generator now use 'generated' for output-only tokens
        sequences = []
        for seq in data['generated']:
            if torch.is_tensor(seq):
                if seq.dim() == 2 and seq.shape[1] == 5:
                    # Extract just the token indices (first column)
                    sequences.append(seq[:, 0])
                elif seq.dim() == 1:
                    sequences.append(seq)
                else:
                    # Unknown format, try to use as is
                    sequences.append(seq)
            else:
                sequences.append(torch.tensor(seq))
        
        # Determine format based on presence of 'trajectories' key
        format_type = 'trajectory' if 'trajectories' in data else 'sample'
        return sequences, format_type
    
    else:
        raise ValueError("Unknown file format - no 'generated' key found")


def compare_sequences_detailed(seq1: torch.Tensor, seq2: torch.Tensor, max_diffs: int = 10) -> Dict[str, Any]:
    """Compare two sequences token by token."""
    min_len = min(len(seq1), len(seq2))
    
    # Handle empty sequences
    if min_len == 0:
        return {
            'len1': len(seq1),
            'len2': len(seq2),
            'compared_length': 0,
            'num_matches': 0,
            'num_diffs': 0,
            'match_rate': 0,
            'first_diff_idx': None,
            'diff_positions': []
        }
    
    # Convert to numpy for easier comparison
    arr1 = seq1[:min_len].numpy() if torch.is_tensor(seq1) else seq1[:min_len]
    arr2 = seq2[:min_len].numpy() if torch.is_tensor(seq2) else seq2[:min_len]
    
    # Ensure both are 1D arrays for comparison
    if arr1.ndim > 1:
        arr1 = arr1.flatten()
    if arr2.ndim > 1:
        arr2 = arr2.flatten()
    
    # Find differences
    try:
        matches = (arr1 == arr2)
    except ValueError as e:
        # Shape mismatch - should not happen after flattening but just in case
        print(f"Warning: Shape mismatch - seq1 shape {arr1.shape}, seq2 shape {arr2.shape}")
        return {
            'len1': len(seq1),
            'len2': len(seq2),
            'compared_length': min_len,
            'num_matches': 0,
            'num_diffs': min_len,
            'match_rate': 0,
            'first_diff_idx': 0,
            'diff_positions': [{'position': 0, 'error': 'Shape mismatch'}]
        }
    
    num_matches = np.sum(matches)
    num_diffs = min_len - num_matches
    
    # Find first difference
    first_diff_idx = None
    diff_positions = []
    
    for i in range(min_len):
        if not matches[i]:
            if first_diff_idx is None:
                first_diff_idx = i
            if len(diff_positions) < max_diffs:
                diff_positions.append({
                    'position': i,
                    'val1': int(arr1[i]),
                    'val2': int(arr2[i])
                })
    
    return {
        'len1': len(seq1),
        'len2': len(seq2),
        'compared_length': min_len,
        'num_matches': num_matches,
        'num_diffs': num_diffs,
        'match_rate': num_matches / min_len if min_len > 0 else 0,
        'first_diff_idx': first_diff_idx,
        'diff_positions': diff_positions
    }


def compare_all_sequences(data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare all sequences between two files."""
    # Now we're comparing ONLY output tokens since 'generated' contains only outputs
    output_seqs1, format1 = extract_token_sequences(data1)
    output_seqs2, format2 = extract_token_sequences(data2)
    
    # Get input sequences if available (for reporting)
    inputs1 = data1.get('inputs', [])
    inputs2 = data2.get('inputs', [])
    
    num_seqs1 = len(output_seqs1)
    num_seqs2 = len(output_seqs2)
    num_to_compare = min(num_seqs1, num_seqs2)
    
    comparison_results = []
    total_matches = 0
    total_tokens = 0
    
    # We're ONLY comparing outputs now
    for i in range(num_to_compare):
        result = compare_sequences_detailed(output_seqs1[i], output_seqs2[i])
        comparison_results.append(result)
        total_matches += result['num_matches']
        total_tokens += result['compared_length']
    
    # Calculate input statistics separately (if we want to report them)
    input_stats = None
    if inputs1 and inputs2:
        input_matches = 0
        input_tokens = 0
        for i in range(min(len(inputs1), len(inputs2))):
            # Compare input sequences
            inp1 = inputs1[i]
            inp2 = inputs2[i]
            if torch.is_tensor(inp1) and inp1.dim() == 2:
                inp1 = inp1[:, 0]  # Extract token indices
            if torch.is_tensor(inp2) and inp2.dim() == 2:
                inp2 = inp2[:, 0]  # Extract token indices
            
            if torch.is_tensor(inp1):
                inp1 = inp1.numpy()
            if torch.is_tensor(inp2):
                inp2 = inp2.numpy()
            
            min_len = min(len(inp1) if hasattr(inp1, '__len__') else 0,
                         len(inp2) if hasattr(inp2, '__len__') else 0)
            if min_len > 0:
                input_tokens += min_len
                # Inputs should always match if from same dataset
                try:
                    matches = np.sum(inp1[:min_len] == inp2[:min_len])
                    input_matches += matches
                except:
                    pass  # Skip if comparison fails
        
        input_stats = {
            'tokens': input_tokens,
            'matches': input_matches,
            'match_rate': input_matches / input_tokens if input_tokens > 0 else 0
        }
    
    return {
        'format1': format1,
        'format2': format2,
        'num_sequences1': num_seqs1,
        'num_sequences2': num_seqs2,
        'num_compared': num_to_compare,
        'total_output_tokens_compared': total_tokens,  # Renamed to be clear
        'total_output_matches': total_matches,  # Renamed to be clear
        'output_match_rate': total_matches / total_tokens if total_tokens > 0 else 0,
        'sequence_comparisons': comparison_results,
        'input_stats': input_stats,  # Separate input comparison (should be 100% if same dataset)
    }


def print_comparison_report(comparison: Dict[str, Any], file1: Path, file2: Path, verbose: bool = False):
    """Print a detailed comparison report."""
    print("\n" + "=" * 80)
    print("INFERENCE OUTPUT COMPARISON REPORT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("FILES COMPARED:")
    print(f"  File 1 ({comparison['format1']}): {file1}")
    print(f"  File 2 ({comparison['format2']}): {file2}")
    print()
    
    # Show sample outputs for first few sequences to understand what's happening
    print("SAMPLE OUTPUT COMPARISON (First 5 sequences):")
    print("-" * 40)
    for i in range(min(5, comparison['num_compared'])):
        result = comparison['sequence_comparisons'][i]
        print(f"\nSequence {i}:")
        print(f"  File 1 length: {result['len1']:4d} tokens")
        print(f"  File 2 length: {result['len2']:4d} tokens")
        
        # Load actual sequences to show them
        data1 = torch.load(file1, map_location='cpu')
        data2 = torch.load(file2, map_location='cpu')
        
        if 'generated' in data1 and 'generated' in data2:
            seq1 = data1['generated'][i] if i < len(data1['generated']) else None
            seq2 = data2['generated'][i] if i < len(data2['generated']) else None
            
            if seq1 is not None and seq2 is not None:
                # Extract token values
                tokens1 = seq1[:, 0] if seq1.dim() == 2 else seq1
                tokens2 = seq2[:, 0] if seq2.dim() == 2 else seq2
                
                # Show first 10 tokens from each
                show_len = 10
                print(f"  File 1 first {min(show_len, len(tokens1))} tokens: {tokens1[:show_len].tolist()}")
                print(f"  File 2 first {min(show_len, len(tokens2))} tokens: {tokens2[:show_len].tolist()}")
                
                if result['len1'] != result['len2']:
                    print(f"  ⚠️  LENGTH MISMATCH: Only compared first {result['compared_length']} tokens!")
                
                if result['num_diffs'] > 0:
                    print(f"  ❌ DIFFERENCES: {result['num_diffs']} mismatches in compared portion")
                elif result['compared_length'] > 0:
                    print(f"  ✓ Compared portion matches ({result['compared_length']} tokens)")
    print()
    
    # Check if we have output-only comparison
    has_output_comparison = comparison.get('output_comparison') and comparison['output_comparison'].get('available')
    
    if has_output_comparison:
        print("=" * 80)
        print("GENERATED OUTPUT COMPARISON (Excluding Input Tokens)")
        print("=" * 80)
        output_comp = comparison['output_comparison']
        print(f"  Total output tokens compared: {output_comp['total_tokens_compared']:,}")
        print(f"  Total matching output tokens: {output_comp['total_matches']:,}")
        print(f"  Output match rate: {output_comp['overall_match_rate']:.2%}")
        
        # Count perfect matches in outputs
        perfect_output_matches = sum(1 for r in output_comp['sequence_comparisons'] if r['match_rate'] == 1.0)
        sequences_with_output_diffs = sum(1 for r in output_comp['sequence_comparisons'] if r['num_diffs'] > 0)
        
        print(f"  Perfect output matches: {perfect_output_matches}/{len(output_comp['sequence_comparisons'])} ({perfect_output_matches/len(output_comp['sequence_comparisons'])*100:.1f}%)")
        print(f"  Sequences with output differences: {sequences_with_output_diffs}/{len(output_comp['sequence_comparisons'])} ({sequences_with_output_diffs/len(output_comp['sequence_comparisons'])*100:.1f}%)")
        print()
    
    print("SUMMARY STATISTICS:")
    print("-" * 40)
    print(f"  Total sequences compared: {comparison['num_compared']}")
    
    # Length statistics
    length_mismatches = sum(1 for r in comparison['sequence_comparisons'] if r['len1'] != r['len2'])
    if length_mismatches > 0:
        print(f"  ⚠️ Sequences with length mismatches: {length_mismatches}/{comparison['num_compared']} ({length_mismatches/comparison['num_compared']*100:.1f}%)")
        
        # Show average lengths
        avg_len1 = np.mean([r['len1'] for r in comparison['sequence_comparisons']])
        avg_len2 = np.mean([r['len2'] for r in comparison['sequence_comparisons']])
        print(f"     Average length File 1: {avg_len1:.0f} tokens")
        print(f"     Average length File 2: {avg_len2:.0f} tokens")
    
    # Token comparison (only overlapping portions)
    print()
    print(f"  Overlapping token comparison:")
    print(f"     Total tokens compared: {comparison.get('total_output_tokens_compared', 0):,}")
    print(f"     Matching tokens: {comparison.get('total_output_matches', 0):,}")
    print(f"     Match rate: {comparison.get('output_match_rate', 0):.2%}")
    
    # Show input comparison if available (should always match)
    if comparison.get('input_stats'):
        print()
        print(f"  Input verification:")
        input_stats = comparison['input_stats']
        if input_stats['match_rate'] == 1.0:
            print(f"     ✓ Inputs match (same dataset)")
        else:
            print(f"     ❌ INPUT MISMATCH: Only {input_stats['match_rate']:.2%} match")
    print()
    
    # Find sequences with perfect matches
    perfect_matches = sum(1 for r in comparison['sequence_comparisons'] if r['match_rate'] == 1.0)
    print(f"  Perfect sequence matches: {perfect_matches}/{comparison['num_compared']} ({perfect_matches/comparison['num_compared']*100:.1f}%)")
    
    # Find sequences with any differences
    sequences_with_diffs = sum(1 for r in comparison['sequence_comparisons'] if r['num_diffs'] > 0)
    print(f"  Sequences with differences: {sequences_with_diffs}/{comparison['num_compared']} ({sequences_with_diffs/comparison['num_compared']*100:.1f}%)")
    print()
    
    if verbose:
        print("DETAILED SEQUENCE COMPARISON:")
        print("-" * 40)
        
        for i, result in enumerate(comparison['sequence_comparisons']):
            if result['num_diffs'] > 0:  # Only show sequences with differences
                print(f"\nSequence {i}:")
                print(f"  Length: {result['len1']} vs {result['len2']}")
                print(f"  Matches: {result['num_matches']}/{result['compared_length']} ({result['match_rate']:.1%})")
                print(f"  First difference at position: {result['first_diff_idx']}")
                
                if result['diff_positions']:
                    print(f"  First {len(result['diff_positions'])} differences:")
                    for diff in result['diff_positions']:
                        print(f"    Position {diff['position']}: {diff['val1']} vs {diff['val2']}")
    
    # Check for length mismatches
    length_mismatches = sum(1 for r in comparison['sequence_comparisons'] if r['len1'] != r['len2'])
    if length_mismatches > 0:
        print("LENGTH MISMATCHES:")
        print("-" * 40)
        print(f"  Sequences with different lengths: {length_mismatches}/{comparison['num_compared']}")
        
        if verbose:
            for i, result in enumerate(comparison['sequence_comparisons']):
                if result['len1'] != result['len2']:
                    print(f"    Sequence {i}: {result['len1']} vs {result['len2']} tokens")
        print()
    
    # Analyze where differences occur
    if sequences_with_diffs > 0:
        first_diff_positions = [r['first_diff_idx'] for r in comparison['sequence_comparisons'] if r['first_diff_idx'] is not None]
        if first_diff_positions:
            avg_first_diff = np.mean(first_diff_positions)
            median_first_diff = np.median(first_diff_positions)
            
            print("DIFFERENCE ANALYSIS:")
            print("-" * 40)
            print(f"  Average position of first difference: {avg_first_diff:.1f}")
            print(f"  Median position of first difference: {median_first_diff:.1f}")
            print(f"  Earliest difference at position: {min(first_diff_positions)}")
            print(f"  Latest difference at position: {max(first_diff_positions)}")
            print()
    
    print("=" * 80)
    
    # Final verdict based on OUTPUT comparison (since we now compare outputs only)
    output_match_rate = comparison.get('output_match_rate', 0)
    
    # Check for length mismatches
    length_mismatches = sum(1 for r in comparison['sequence_comparisons'] if r['len1'] != r['len2'])
    length_mismatch_rate = length_mismatches / comparison['num_compared'] if comparison['num_compared'] > 0 else 0
    
    print("VERDICT:")
    print("-" * 40)
    
    # First report on length mismatches
    if length_mismatch_rate > 0.5:
        print(f"⚠️  WARNING: {length_mismatches}/{comparison['num_compared']} sequences have different lengths!")
        print(f"   Only the overlapping portion was compared.")
        
        # Calculate average length ratio
        length_ratios = []
        for r in comparison['sequence_comparisons']:
            if r['len1'] > 0 and r['len2'] > 0:
                ratio = min(r['len1'], r['len2']) / max(r['len1'], r['len2'])
                length_ratios.append(ratio)
        if length_ratios:
            avg_ratio = np.mean(length_ratios)
            print(f"   Average length similarity: {avg_ratio:.1%}")
        print()
    
    # Then report on token matching in overlapping portions
    if output_match_rate == 1.0 and length_mismatch_rate == 0:
        print("✅ PERFECT MATCH: All output tokens are identical!")
    elif output_match_rate == 1.0 and length_mismatch_rate > 0:
        print("⚠️  PARTIAL MATCH: Overlapping portions match perfectly, but lengths differ significantly")
    elif output_match_rate > 0.95:
        print(f"✓ GOOD MATCH: {output_match_rate:.2%} of compared output tokens match")
    elif output_match_rate > 0.80:
        print(f"⚠ MODERATE MATCH: {output_match_rate:.2%} of compared output tokens match")
    else:
        print(f"❌ POOR MATCH: Only {output_match_rate:.2%} of compared output tokens match")
    
    print("=" * 80)


def save_comparison_report(comparison: Dict[str, Any], output_path: Path):
    """Save the comparison results to a file."""
    output_data = {
        'comparison': comparison,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(output_data, output_path)
    print(f"\nComparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare two inference output files')
    parser.add_argument('file1', type=str, help='First .pt file to compare')
    parser.add_argument('file2', type=str, help='Second .pt file to compare')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed sequence-by-sequence comparison')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save comparison report to file')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum number of sequences to compare')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load files
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    logging.info(f"Loading {file1_path}...")
    data1 = load_inference_file(file1_path)
    
    logging.info(f"Loading {file2_path}...")
    data2 = load_inference_file(file2_path)
    
    # Compare sequences
    logging.info("Comparing sequences...")
    comparison = compare_all_sequences(data1, data2)
    
    # Limit comparison if requested
    if args.max_sequences:
        comparison['sequence_comparisons'] = comparison['sequence_comparisons'][:args.max_sequences]
        comparison['num_compared'] = min(args.max_sequences, comparison['num_compared'])
    
    # Print report
    print_comparison_report(comparison, file1_path, file2_path, verbose=args.verbose)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        save_comparison_report(comparison, output_path)


if __name__ == "__main__":
    main()