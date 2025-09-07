import numpy as np
import random

import torch
import os

from typing import List

from src.token import SpecialToken

def dump_input(input_ids: torch.Tensor, indices, prefix="", dump_dir='./temp'):
    """
    Dump input tensors to files for debugging or analysis.
    
    Args:
    input_ids (torch.Tensor): The input tensor containing token IDs.
    prefix (str): A prefix for the filename.
    dump_dir (str): Directory to save the dumps.
    """
    
    # Save input_ids
    torch.save({
            'input_ids': input_ids,
            'indices': indices
        }, os.path.join(dump_dir, f"{prefix}_input_ids.pt"))

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

def create_mask(input_ids, device, end_of_example_indices:List[int], mask_hack=False):
    assert input_ids.dim() == 3
    batch_size, seq_length, _ = input_ids.shape
    assert batch_size == len(end_of_example_indices)

    mask = torch.triu(torch.ones((seq_length, seq_length), device=device), diagonal=1).bool()

    # mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # NOT PROVEN YET: was a bug, causing unpredictable logits outputs!
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # it will later be expand again by number of heads in the transformer, this is also required for DataParallel

    if mask_hack:
        for batch_index, end_index in enumerate(end_of_example_indices):
            assert end_index > 0
            # assert end_index <= seq_length
            # print('batch_index, end_index', batch_index, end_index)
            mask[batch_index, :, :end_index] = 0

    return mask

def prefix_mask(target, seq_length, end_of_examples:List[int]):
    prefix_length = torch.tensor(end_of_examples, device = target.device)    
    mask = torch.arange(seq_length, device=target.device).unsqueeze(0) < prefix_length.unsqueeze(1) - 1
    target[mask] = SpecialToken.PAD.value
    
    return target

def mask_expansion(mask, num_heads):
    assert mask.dim() == 3
    batch_size, sequence_length, _ = mask.shape
    mask = mask.unsqueeze(1)  # [batch_size, 1, sequence_length, sequence_length]
    mask = mask.expand(-1, num_heads, -1, -1)  # [batch_size, heads, sequence_length, sequence_length]
    mask = mask.reshape(num_heads * batch_size, sequence_length, sequence_length)  # e.g. Final shape: [2800, 203, 203]

    assert mask.dtype is torch.bool, f"mask.dtype is not bool, {mask.dtype}"
    return mask

def combine_encoding(x: torch.Tensor, batch_size: int, seq_length: int, max_grid_size: int, grid_scale: torch.Tensor, grid_encoding: torch.Tensor):
    # Always use the optimized grid encoding
    # -1 will access the last element so, it is OK
    # Assert that all values are within the expected range
    assert torch.all(x[:, :, 1:] >= -1), "Indices out of range"
    assert torch.all(x[:, :, 1:] < max_grid_size), "Indices out of range"

    row_encodings = grid_encoding[:, x[:, :, 1], :]
    col_encodings = grid_encoding[:, x[:, :, 2], :]
    top_encodings = grid_encoding[:, x[:, :, 3], :]
    sample_encoding = grid_encoding[:, x[:, :, 4], :]
    grid_encodings = torch.cat([
        row_encodings * grid_scale[0],
        col_encodings * grid_scale[1],
        top_encodings * grid_scale[2],
        sample_encoding
    ], dim=-1)  # (N, seq_length, embed_size // 2)

    return grid_encodings.squeeze(0)

def count_parameters(model, save_dir=None):
    def format_number(num):
        """Format number with human-readable suffix"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    output_lines = []
    output_lines.append("Layer_name\t\t\t\t\tShape\t\t\tParameters\tReadable")
    output_lines.append("="*100)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        shape_str = str(list(parameter.shape))
        # Pad name to ensure alignment
        name_padded = name.ljust(40)
        shape_padded = shape_str.ljust(20)
        output_lines.append(f"{name_padded}\t{shape_padded}\t{param}\t\t{format_number(param)}")
        total_params += param
    output_lines.append("="*100)
    output_lines.append(f"Total Trainable Parameters: {total_params:,} ({format_number(total_params)})")
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to file if save_dir provided
    if save_dir:
        with open(os.path.join(save_dir, 'model_parameters.txt'), 'w') as f:
            f.write('\n'.join(output_lines))
    
    return total_params

def dump_model_operation(outputs, input_ids, mask, file_name):
    torch.save({'outputs': outputs, 'input_ids': input_ids, 'mask': mask}, f"{file_name}.pt")