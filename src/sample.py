import logging
import argparse
import torch
from torch.amp import autocast
from src.model import Transformer
from src.token import VOCAB_SIZE, SpecialToken
from src.load_data import load_from_json, GridDataset
from src.utils.helper import set_deterministic
from src.utils.display_diff import compare_sequences, colorize, split_into_chunks
# Mask creation no longer needed - model uses causal attention internally
from src.checkpoint_handler import CheckpointHandler
from src.utils.logger_helper import setup_logging
from src.inference_output_saver import InferenceOutputSaver
from pathlib import Path
from datetime import datetime

debug_on_cpu = False

def format_batch(batch, max_print_length=150):
    def token_to_str(token):
        if token < SpecialToken.CELL_TOKEN_SIZE.value:
            return str(token)
        return SpecialToken(token).name

    formatted_sequences = []
    for sequence in batch:
        tokens = [token_to_str(t.item()) for t in sequence[:max_print_length]]
        if len(sequence) > max_print_length:
            tokens.append('...')
        formatted_sequences.append(' '.join(tokens))
    
    return '\n\n'.join(formatted_sequences)

def generate_sample(model, input_sequence, max_length, device, *, early_stop = None):
    model.eval()
    y = 0
    x = 0
    coord = (-1, -1)
    
    with torch.no_grad():
        input_ids = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_length)

        seq_length = len(input_sequence)
        for generated_token_index in range(max_length - seq_length):
            # Use autocast for bfloat16 inference
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # Model now expects only input_ids with shape (batch, seq_len, 5)
                outputs = model(input_ids)  # (1, seq_length, vocab_size)
            next_token_logits = outputs[0, -1, :].float()  # Convert back to float32 for token selection
            next_token_id = torch.argmax(next_token_logits).item()

            if early_stop:
                expected_token = early_stop[seq_length + generated_token_index]
                match = "✓" if expected_token == next_token_id else "✗"
                print(f'\r[{generated_token_index:3d}] gen:{next_token_id:3d} exp:{expected_token:3d} {match}', end="", flush=True)
                
                if expected_token != next_token_id:
                    print()  # New line before returning on failure
                    # Return the generated sequence so far, not an empty list!
                    return input_ids.squeeze(0).tolist(), generated_token_index    # Do not put + 1 here, because we are counting the number of correct tokens
            else:
                print(f'\r{generated_token_index} ', end="", flush=True)

            if next_token_id < SpecialToken.CELL_TOKEN_SIZE.value:
                coord = (y, x)
                x = x + 1
                x = min(x, model.max_grid_size - 1)
            elif next_token_id == SpecialToken.ROW_SEPARATOR.value:
                coord = (y, x)
                x = 0
                y = y + 1
                y = min(y, model.max_grid_size - 1)
            else:
                y = 0
                x = 0
                coord = (-1, -1)

            input_ids = torch.cat([input_ids, torch.tensor([[[next_token_id, coord[0], coord[1], -1, -1]]], dtype=torch.long, device=device)], dim=1)  # (1, seq_length + 1)
            if next_token_id == SpecialToken.END.value or \
                (generated_token_index > 1000 and not early_stop): # we don't know the right answer and this has generated more than 30x30 cells
                return input_ids.squeeze(0).tolist(), generated_token_index + 1
            torch.cuda.empty_cache()
            
    return input_ids.squeeze(0).tolist(), max_length - seq_length

def main():
    parser = argparse.ArgumentParser(description='Generate samples using the trained model')
    
    parser.add_argument('--data-source', type=str, default='arc-agi_evaluation',
                        help='Data source to use (default: arc-agi_evaluation)')
    
    parser.add_argument('--checkpoint-path', type=str, default='cloud_runs/69.55.141.236/2500/runs/2500/20241029_043432_nogit_nobranch_lr3e-05_bl2e-05_ssu0_bs21_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_mh0_ssnone_ss1e-02_c11/Transformer_latest.pt',
                        help='Path to the checkpoint file')

    parser.add_argument('--start-testing-index', type=int, default=0,
                        help='Index of the sample to generate (default: 0)')

    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    
    parser.add_argument('--second-only', action='store_true', default=False,
                        help='Use second test only')
    
    parser.add_argument('--unlimited-sequence', action='store_true', default=True,
                        help='Index of the max sequence to be inf')
    
    parser.add_argument('--logger-file', type=str, default='sample_generation.log',
                    help='Log file name (default: sample_generation.log)')
    
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output .pt file to save generated sequences (default: tmp/sample_output_<timestamp>.pt)')
    

    args = parser.parse_args()

    setup_logging(args.logger_file)  # Set up logging
    

    set_deterministic()

    # Load data
    data_sources = [args.data_source]
    all_challenges = {}

    for source in data_sources:
        try:
            challenges, solutions = load_from_json(source, './input_data/')
            all_challenges.update(challenges)
        except FileNotFoundError as e:
            logging.error("Error loading %s: %s. Skipping this data source.", source, e)

    if not all_challenges:
        logging.error("No data could be loaded. Please check the file paths and data sources.")
        return

    device = torch.device("cuda" if (torch.cuda.is_available() and not debug_on_cpu) else "cpu")
    model, max_seq_length, checkpoint_args, checkpoint_info = CheckpointHandler.load_checkpoint_in_production(args.checkpoint_path, device, adjust_max_length=12000 if args.unlimited_sequence else 0)
    
    # Model stays in FP32 - autocast will handle BF16 conversion during inference
    # This is the proper way to use mixed precision

    # Mask handling is now done internally by the model with causal attention

    # Create the dataset
    dataset_ref = GridDataset.load_from_paired_file(all_challenges, solutions, second_only=args.second_only)
    dataset = GridDataset.load_from_paired_file(all_challenges, None, second_only=args.second_only)

    logging.info(f'Processing: {args.checkpoint_path}')

    logging.info('dataset: %s %d', args.data_source, len(dataset))
    logging.info('dataset_ref: %d', len(dataset_ref))
    logging.info('max_seq_length: %d', max_seq_length)

    mismatch_count = 0    
    oom_count = 0
    tested_count = 0
    total_correct_token_length = 0
    
    # Create output saver to collect samples
    output_saver = InferenceOutputSaver(args.output_file)
    
    # Generate samples
    start_index = args.start_testing_index
    for i in range(start_index, len(dataset)):
        input_sequence = dataset[i]
        # print('input_sequence a', input_sequence)
        expected_sequence = [s[0] for s in dataset_ref[i]['task']]
        
        generated_sample = None
        generated_length = 0
        finished = False
        was_tested = False
    
        if max_seq_length > len(input_sequence['task']) and max_seq_length > len(expected_sequence):
            tested_count += 1
            was_tested = True
            
            try:
                sample, generated_length = generate_sample(model, input_sequence['task'], max_seq_length, device, early_stop = expected_sequence)
                total_correct_token_length += generated_length
                

                sample_tokens = [s[0] for s in sample]
                generated_sample = sample  # Keep full 5D format for saving
                
                # Check if sequences match
                matches = sample_tokens == expected_sequence[:max_seq_length]
                
                if not matches:
                    mismatch_count += 1
                    finished = False  # Failed - stopped early due to mismatch

                    if args.verbose:
                        input_length = len(input_sequence['task'])
                        logging.info("\nInput sequence (%d, %d):", i, input_length)
                        logging.info(format_batch([torch.tensor(input_sequence['task'])], max_print_length=99999))

                        compare_sequences(
                            format_batch([torch.tensor(expected_sequence[input_length + 1:max_seq_length])]), 
                                        format_batch([torch.tensor(sample_tokens[input_length + 1:])]))
                else:
                    finished = True  # Successfully completed
                    
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                generated_sample = input_sequence['task']  # Just save input on OOM
                finished = False
            except Exception as e:
                logging.error(f"Unexpected error during sample generation at index {i}: {str(e)}")
                mismatch_count += 1  # Count errors as mismatches
                generated_sample = input_sequence['task']  # Just save input on error
                finished = False

            print(f' ____. Tested {tested_count}@{i + 1}, generated {generated_length or 0}. Total failed cases = {mismatch_count}/{tested_count}, oom_count: {oom_count}')
        else:
            # Sequence too long for max_seq_length - still need to save for comparison
            generated_sample = input_sequence['task']  # Use input as placeholder
            
        # Collect sequences for saving
        if was_tested:
            # Add sample to output saver
            output_saver.add_sample(
                generated_sample=generated_sample,
                input_sequence=input_sequence,
                finished=finished,
                sample_index=i,
                generated_length=generated_length,
                matches=finished and tested_count > 0 and mismatch_count == 0
            )
            
    avg_correct_per_test = total_correct_token_length / tested_count if tested_count > 0 else 0
    total_tokens_generated = total_correct_token_length + mismatch_count  # Each failed test contributes 1 wrong token
    
    logging.info('\n' + '=' * 50)
    logging.info('EVALUATION RESULTS')
    logging.info('=' * 50)
    logging.info('Checkpoint: %s', args.checkpoint_path)
    logging.info('Epoch: %d, Train Loss: %.6f, Val Loss: %.6f', 
                 checkpoint_info['epoch'], checkpoint_info['train_loss'], checkpoint_info['val_loss'])
    logging.info('Dataset: %s', args.data_source)
    logging.info('-' * 50)
    logging.info('Failed cases = %d/%d, success rate %.2f%%', mismatch_count, tested_count, (tested_count - mismatch_count) * 100. / tested_count)
    logging.info('Token statistics:')
    logging.info('  - Total correct tokens: %d', total_correct_token_length)
    logging.info('  - Total wrong tokens: %d (one per failed test - causes early stopping)', mismatch_count)
    logging.info('  - Total tokens generated: %d', total_tokens_generated)
    logging.info('  - Average correct tokens per test before failure: %.1f', avg_correct_per_test)
    logging.info('=' * 50)
    
    # Save all collected samples
    metadata = {
        'checkpoint': args.checkpoint_path,
        'dataset': args.data_source,
        'temperature': 0.0,  # src.sample doesn't use temperature
        'tested_count': tested_count,
        'mismatch_count': mismatch_count,
        'success_rate': (tested_count - mismatch_count) * 100.0 / tested_count if tested_count > 0 else 0,
        'total_correct_tokens': total_correct_token_length,
        'avg_correct_per_test': avg_correct_per_test,
        'timestamp': datetime.now().isoformat(),
        'second_only': args.second_only,
        'checkpoint_info': checkpoint_info
    }
    
    output_saver.save(metadata)

if __name__ == "__main__":
    main()
