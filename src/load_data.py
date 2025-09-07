import json

import torch
from torch.utils.data import Dataset

from typing import Dict, List, Tuple, Any, Set, List

from src.token import SpecialToken
from src.utils.data_augment import augment_compact_grids
from src.utils.helper import print_thread_info
from src.utils.grid_data_process import shuffle_all_but_last_pair, tokenize_compact_task, end_of_examples_mark, preprocess_for_fast_access, preprocess_array_data, pad_collate, shuffle_all_pairs, create_special_token_sequence
class GridDataset(Dataset):
    def __init__(self):
        self.data = []
        self.augment_seed = -1
        self.current_attempt_length = 0  # Current max attempt tokens from scheduler
        self.trajectory_data = None  # Optional: trajectory data with attempt tokens

    def __len__(self):
        return len(self.data)

    @staticmethod
    def convert_to_int_lists(np_data):
        return [int(x) for x in np_data]
    
    def insert_attempt_section(self, task, idx):
        """
        Insert attempt section into the task.
        
        For original data:
        - Insert empty attempt section: [ATTEMPT_START][ATTEMPT_END]
        
        For trajectory data:
        - Insert actual attempt tokens between [ATTEMPT_START] and [ATTEMPT_END]
        - Truncate to current_attempt_length if necessary
        
        Args:
            task: Tokenized task data
            idx: Sample index
            
        Returns:
            Modified task with attempt section inserted
        """
        # Find where to insert the attempt section
        # It should be after the input, before the output
        end_of_examples_index = end_of_examples_mark(task)
        
        # Check if this is trajectory data with attempt tokens
        if self.trajectory_data and idx in self.trajectory_data:
            # Get the attempt tokens for this sample
            attempt_tokens = self.trajectory_data[idx].get('attempt_tokens', [])
            
            # Truncate to current max attempt length if needed
            if self.current_attempt_length > 0 and len(attempt_tokens) > self.current_attempt_length:
                attempt_tokens = attempt_tokens[:self.current_attempt_length]
            
            # Insert [ATTEMPT_START] + attempt_tokens + [ATTEMPT_END]
            attempt_section = [
                create_special_token_sequence(SpecialToken.ATTEMPT_START, 0),
                *attempt_tokens,
                create_special_token_sequence(SpecialToken.ATTEMPT_END, 0)
            ]
        else:
            # Original data: insert empty attempt section
            # When attempt_length is 0, just use ATTEMPT_START and ATTEMPT_END
            # When attempt_length > 0, we could add PAD tokens or dummy tokens
            # But per design: "No PAD tokens in attempt sections"
            # So we keep it empty regardless of attempt_length for original data
            attempt_section = [
                create_special_token_sequence(SpecialToken.ATTEMPT_START, 0),
                create_special_token_sequence(SpecialToken.ATTEMPT_END, 0)
            ]
            # Note: current_attempt_length affects trajectory data only
            # Original data always gets empty attempt sections
        
        # Insert the attempt section after the input, before the output
        # The task format is: [examples] [input] [output]
        # We want: [examples] [input] [attempt_section] [output]
        modified_task = task[:end_of_examples_index] + attempt_section + task[end_of_examples_index:]
        
        return modified_task

    def __getitem__(self, idx):
        if self.augment_seed >= 0:                
            augmented_task = augment_compact_grids(self.convert_to_int_lists(self.data[idx]))
            shuffled_task = shuffle_all_but_last_pair(augmented_task)
            task = tokenize_compact_task(shuffled_task)
        else:
            task = tokenize_compact_task(self.convert_to_int_lists(self.data[idx]))
        
        # Add attempt section to the task
        task = self.insert_attempt_section(task, idx)
        
        # Recalculate end_of_examples_index after adding attempt section
        end_of_examples_index = end_of_examples_mark(task)
        assert end_of_examples_index > 0
        
        return {
            'task': task,
            'idx': idx,
            'end_of_examples_index': end_of_examples_index
        }

    def set_augment_seed(self, augment_seed):
        self.augment_seed = augment_seed
    
    def set_max_length(self, max_length):
        self.max_length = int(max_length)
    
    def set_attempt_length(self, attempt_length):
        """Set the current maximum attempt length from the scheduler"""
        self.current_attempt_length = int(attempt_length)
        
        # Log what this means for the data
        if self.trajectory_data:
            trajectory_count = len(self.trajectory_data)
            # Trajectory data will be truncated to this length
            effective_msg = f"(will truncate {trajectory_count} trajectory samples)"
        else:
            # Original data always gets empty attempt sections
            effective_msg = "(no trajectory data - empty attempt sections only)"
        
        import logging
        logging.debug(f"GridDataset: attempt_length set to {self.current_attempt_length} {effective_msg}")

    def set_source_ranges(self, source_ranges: Dict[str, Tuple[int, int]]):
        self.source_ranges = source_ranges

    def cut_long_sequence(self, threshold_length):
        # iterate over self.data remove all elements longer than threshold_length
        self.data = [seq for seq in self.data if len(seq) <= threshold_length]

    def sort_by_length(self, *, reverse:bool):
        self.data.sort(key=len, reverse=reverse)

    @classmethod
    def load_from_paired_file(cls, challenges: Dict[str, Any], solutions: Dict[str, Any], source_ranges: Dict[str, Tuple[int, int]] = {'ignore': (-1, -1)}, second_only: bool = False) -> 'GridDataset':
        instance = cls()
        instance.source_ranges = source_ranges
        instance.second_only = second_only
        preprocess_for_fast_access(challenges, solutions, instance.second_only, instance.data)
        return instance
            
    def pad_collate(self, batch):
        return pad_collate(batch, self.max_length)

class DynamicGridDataset(Dataset):
    def __init__(self, compact_grid, sample_size, max_seq_length):
        self.compact_grid = compact_grid
        self.sample_size = sample_size
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        augmented_task = augment_compact_grids(self.compact_grid)
        shuffled_task = shuffle_all_pairs(augmented_task)
        task = tokenize_compact_task(shuffled_task)
        
        end_of_examples_index = end_of_examples_mark(task)
        assert end_of_examples_index > 0
        
        return {
            'task': task,
            'idx': idx,
            'end_of_examples_index': end_of_examples_index
        }
        
    def pad_collate(self, batch):
        return pad_collate(batch, self.max_seq_length)


class MixedGridDataset(Dataset):
    """
    Mixed dataset that combines original data with trajectory data for pseudo-RL training
    """
    def __init__(self, original_dataset: GridDataset, trajectory_data: Dict[int, Dict[str, Any]]):
        """
        Initialize mixed dataset
        
        Args:
            original_dataset: Original GridDataset
            trajectory_data: Dictionary mapping sample indices to trajectory data
                           (ALL trajectories will be used)
        """
        self.original_dataset = original_dataset
        self.trajectory_data = trajectory_data
        
        # Inherit properties from original dataset
        self.data = original_dataset.data
        self.augment_seed = original_dataset.augment_seed
        self.current_attempt_length = original_dataset.current_attempt_length
        self.max_length = getattr(original_dataset, 'max_length', 2048)
        
        # Use ALL trajectories that we have
        self.use_trajectory = set(trajectory_data.keys())
        
    def __len__(self):
        return len(self.original_dataset)
    
    def set_augment_seed(self, augment_seed):
        self.augment_seed = augment_seed
        self.original_dataset.set_augment_seed(augment_seed)
    
    def set_max_length(self, max_length):
        self.max_length = int(max_length)
        self.original_dataset.set_max_length(max_length)
    
    def set_attempt_length(self, attempt_length):
        """Set the current maximum attempt length from the scheduler"""
        self.current_attempt_length = int(attempt_length)
        self.original_dataset.set_attempt_length(attempt_length)
        
        # Log the effect for mixed dataset
        if self.trajectory_data:
            traj_count = len(self.trajectory_data)
            import logging
            logging.debug(f"MixedGridDataset: attempt_length set to {self.current_attempt_length} "
                         f"({traj_count} samples have trajectories, rest get empty attempts)")
    
    def __getitem__(self, idx):
        # Check if we should use trajectory data for this sample
        if idx in self.use_trajectory and idx in self.trajectory_data:
            # Use trajectory data - pass it to original dataset's trajectory_data
            self.original_dataset.trajectory_data = {idx: self.trajectory_data[idx]}
        else:
            # Use original data (empty attempt section)
            self.original_dataset.trajectory_data = None
        
        # Get the item from original dataset (it will handle the attempt section)
        return self.original_dataset[idx]
    
    def cut_long_sequence(self, max_length):
        """Cut sequences that are too long"""
        self.original_dataset.cut_long_sequence(max_length)
    
    def pad_collate(self, batch):
        return self.original_dataset.pad_collate(batch)


# Loading JSON data
def load_json(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.load(f)
    return data

def load_from_json(case: str, base_path: str) -> tuple:
    with open(base_path + case + '_challenges.json') as f:
        challenges = json.load(f)

    try:
        with open(base_path + case + '_solutions.json') as f:
            solutions = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {case}_solutions: File not found.")
        solutions = None

    return challenges, solutions
