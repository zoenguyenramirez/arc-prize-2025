import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import autocast
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
import sys
from datetime import datetime
import random
import os
import time
import argparse
import subprocess
from src.utils.iterable_helper import custom_cycle  # Replace the import of cycle
from datetime import timedelta
import torch.cuda.nvtx as nvtx
import multiprocessing
from time import sleep
import gc

from src.model import Transformer
from src.token import VOCAB_SIZE, SpecialToken
from src.prepare_data import load_datasets
from src.checkpoint_handler import CheckpointHandler
from src.utils.logger_helper import setup_logger, setup_file_logger
from src.utils.helper import HyperParameters, generate_run_name, set_deterministic, check_existing_hyperparameters, str2bool, log_hyperparameters, validate_parameters, get_git_hash, get_env, round_up_to_multiple, is_gpu_memory_low
from src.schedulars import WarmupCosineLR, UnifiedTrainingScheduler
from src.utils.transformer_helper import dump_input, prefix_mask, count_parameters
from src.utils.data_loader import CustomDataLoader, split_dataset_and_keeping_the_oder, CacheAllDataLoader

debug_single_batch = False
debug_on_cpu = False

def train(model, custom_loader, optimizer, criterion, device, *, scheduler, samples_per_epoch, epoch, logger, accumulation_steps, file_logger):
    file_logger.info(f'\nTraining {epoch}\n')
    nvtx.range_push("Train Function")
    model.train()
    total_loss = 0
    samples_count = 0
    batch_count = 0
    accumulated_samples = 0
    need_to_zero_grad = True

    while samples_count < samples_per_epoch:
        if need_to_zero_grad:
            optimizer.zero_grad()  # Zero gradients at the beginning of accumulation
            need_to_zero_grad = False
        nvtx.range_push(f"batch{batch_count}")
        try:            
            batch = next(custom_loader)
        except StopIteration:
            logger.warning('train(): Data producer is too slow!')
            sleep(1)
            continue

        input_ids = batch['data'].to(device)  # (batch_size, seq_length)
        batch_size, seq_length, _ = input_ids.shape
        target = input_ids.clone()
        target = target[:, 1:, 0]  # (batch_size, seq_length-1)
        input_ids = input_ids[:, :-1, :]  # (batch_size, seq_length-1)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = model(input_ids)  # (batch_size, seq_length-1, vocab_size)
            if epoch % 100 == 0 and batch_count == 0:
                dump_input(input_ids, batch['indices'], f"{epoch:04d}_{batch_count:04d}_train")

            assert seq_length == input_ids[:, :, 0].shape[1] + 1, f"{seq_length} != {input_ids[:, :, 0].shape[1]} + 1"
            target = prefix_mask(target, seq_length - 1, batch['end_of_examples'])
            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), target.reshape(-1))  # (batch_size * (seq_length-1), vocab_size)
            loss = loss / accumulation_steps  # Normalize the loss

        loss.backward()

        accumulated_samples += batch['data'].shape[0]

        if (batch_count + 1) % accumulation_steps == 0:
            optimizer.step()

            need_to_zero_grad = True
            accumulated_samples = 0

            if scheduler:
                scheduler.step() 

        total_loss += loss.item() * accumulation_steps  # Undo the normalization for logging

        file_logger.info(f"B {batch_count} {batch['indices']} {seq_length}")

        samples_count += batch['data'].shape[0]  # Assuming the first element is the data
        batch_count += 1
        
        nvtx.range_pop()

        if debug_single_batch:
            break

    assert accumulated_samples == 0, accumulated_samples

    nvtx.range_pop()
    return total_loss / batch_count, samples_count, batch_count

def validate(model, data_loader, criterion, device, *, epoch, logger, file_logger):
    file_logger.info(f'\nValidation {epoch}\n')
    nvtx.range_push("Validate Function")
    model.eval()
    total_loss = 0
    batch_count = 0
    samples_count = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['data'].to(device)
            batch_size, seq_length, _ = input_ids.shape
            target = input_ids.clone()
            target = target[:, 1:, 0]  # (batch_size, seq_length-1)
            input_ids = input_ids[:, :-1, :]  # (batch_size, seq_length-1)

            outputs = model(input_ids)  # (batch_size, seq_length-1, vocab_size)
            if epoch % 100 == 0 and batch_count == 0:
                dump_input(input_ids, f"{epoch:04d}_{batch_count:04d}_validate")

            assert seq_length == input_ids[:, :, 0].shape[1] + 1, f"{seq_length} != {input_ids[:, :, 0].shape[1]} + 1"
            target = prefix_mask(target, seq_length - 1, batch['end_of_examples'])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), target.reshape(-1))  # (batch_size * (seq_length-1), vocab_size)

            total_loss += loss.item()

            file_logger.info(f"B {batch_count} {batch['indices']} {seq_length}")

            batch_count += 1
            samples_count += batch['data'].shape[0]

            if debug_single_batch:
                break


    nvtx.range_pop()
    return total_loss / batch_count if batch_count > 0 else float('inf')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the Model")
    
    parser.add_argument("--embed-size", type=int, default=36,
                        help="Embedding size")
    parser.add_argument("--num-layers", type=int, default=7,
                        help="Number of layers")
    parser.add_argument("--heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=312,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-3,
                        help="Max learning rate")
    parser.add_argument("--base-lr", type=float, default=1e-5,
                        help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=4096 * 4, # we do not change this often, especialy for hyper parameter tuning
                        help="Samples per epoch")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--log-hyperparameters", action='store_true',
                        help='Log hyperparameters to TensorBoard')
    parser.add_argument("--check-existing-hyperparameters", action='store_true',
                        help='Check existing hyperparameters and exit immediately')
    parser.add_argument("--augment-data", type=str2bool, default=True,
                    help='Augment data on the fly')
    parser.add_argument("--runs-name", help="Specify the name for the runs directory", type=str)
    parser.add_argument("--load-checkpoint", type=str, default=None,
                    help="Path to a checkpoint to load")
    parser.add_argument("--minimize-checkpoints", action='store_true',
                    help="Minimize checkpoints by skipping latest and periodic saves")

    parser.add_argument("--dataset-files", type=str, nargs='+', default=["./intermediate_data/prepared_dataset.pth"],
                    help="Paths to the prepared dataset files")
    
    parser.add_argument("--trajectory-folder", type=str, default=None,
                        help="Optional folder containing trajectory files for pseudo-RL training")

    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of warm-up epochs")
    
    parser.add_argument("--scheduler-config", type=str, default=None,
                        help="Path to scheduler_config.json (required for orchestrated runs)")
    parser.add_argument("--scheduler-previous-state", type=str, default=None,
                        help="Path to previous scheduler_state.json to continue from")

    parser.add_argument("--must-use-cuda", action='store_true', default=True,
                        help="Assert that CUDA must be used for training")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for deterministic behavior")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam'],
                    help="Choose the optimizer: 'sgd' or 'adam'")
    
    # MQA/GQA configuration
    parser.add_argument("--num-kv-heads", type=int, default=1,
                        help="Number of KV heads for MQA/GQA (1=MQA, heads=MHA, between=GQA)")
    
    return parser.parse_args()

def main():
    logger = setup_logger(overwrite_line=False)

    args = parse_arguments()

    logger.info(f'args{args}')

    # Create a HyperParameters instance
    hyperparams = HyperParameters(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        heads=args.heads,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        base_lr=args.base_lr,
        warmup_epochs=args.warmup_epochs,
        augment_data=args.augment_data,
        accumulation_steps=args.accumulation_steps,
        optimizer=args.optimizer,
        num_kv_heads=args.num_kv_heads
    )

    rounded_samples_per_epoch = round_up_to_multiple(args.samples_per_epoch, hyperparams.batch_size * hyperparams.accumulation_steps)

    validate_parameters(hyperparams, args)

    # Generate a unique run name
    run_name = generate_run_name(hyperparams, not bool(args.log_hyperparameters))

    # Create a directory for the current run
    run_dir = f"runs/{args.runs_name}/{run_name}" if args.runs_name else f"runs/{run_name}"

    if args.check_existing_hyperparameters:
        check_existing_hyperparameters(run_dir)

    set_deterministic(seed=args.seed)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() and not debug_single_batch and not debug_on_cpu else "cpu")
    
    if args.must_use_cuda:
        assert device.type == 'cuda', "CUDA is not available, but --must-use-cuda was specified"
    
    model = Transformer(
        VOCAB_SIZE, 
        hyperparams.embed_size, 
        hyperparams.num_layers, 
        hyperparams.heads, 
 
        max_length=args.max_seq_length,
        num_kv_heads=hyperparams.num_kv_heads
    ).to(device)
    
    # Model stays in FP32 - autocast will handle BF16 conversion during computation
    # This ensures optimizer states remain in FP32 for precision
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)


    if hyperparams.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=SpecialToken.PAD.value)

    checkpoint_handler = CheckpointHandler(
        save_dir=run_dir,  # Use the same directory as TensorBoard logs
        minimize_checkpoints=args.minimize_checkpoints
    )
    
    if args.load_checkpoint:
        # TODO: load check point at earlier initialization process (this function)
        checkpoint, best_val_loss = CheckpointHandler.load_checkpoint(args.load_checkpoint, model, device = device, optimizer = optimizer, initial_lr=hyperparams.learning_rate, adjust_max_length = 0)  # grid_encoder is deprecated
        checkpoint_handler.best_val_loss = best_val_loss

        start_epoch = checkpoint['epoch'] + 1
        relay_seed = checkpoint['relay_seed'] + 1

        hyperparams.assert_checkpoint(checkpoint['hyperparameters'])

        run_dir = f'{run_dir}_c{relay_seed}'
        checkpoint_handler.update_save_dir(run_dir)

        # ignore the args
        hyperparams.warmup_epochs = hyperparams.warmup_epochs // 2

        logger.info(f"Loaded checkpoint from {args.load_checkpoint} @ epoch {start_epoch}({checkpoint['epoch_in_session']}) {checkpoint['train_loss'], checkpoint['val_loss']}. run dir is updated to {run_dir}, warmup_epochs is halved")

        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
    else:
        start_epoch = 0
        relay_seed = 0

    os.makedirs(run_dir, exist_ok=True)
    
    # The scheduler config should be created by the orchestrator, not train.py
    # train.py just uses whatever config exists in the run directory
    
    # Count and display model parameters
    count_parameters(model, run_dir)

    # Load the prepared dataset
    dataset, data_sources, _ = load_datasets(args.dataset_files)

    # the dataset was saved by torch, they may not have all attributes
    dataset.set_augment_seed(1 if hyperparams.augment_data else -1)
    dataset.set_max_length(args.max_seq_length) # for pad_collate
    
    # Load trajectories and create mixed dataset if trajectory folder is provided
    if args.trajectory_folder:
        logger.info(f'Loading trajectories from {args.trajectory_folder}')
        from src.trajectory_loader import TrajectoryLoader
        from src.load_data import MixedGridDataset
        
        loader = TrajectoryLoader(args.trajectory_folder)
        trajectory_data = loader.load_trajectories()
        
        if trajectory_data:
            logger.info(f'Creating MixedGridDataset with {len(trajectory_data)} trajectories')
            dataset = MixedGridDataset(
                original_dataset=dataset,
                trajectory_data=trajectory_data
            )
        else:
            logger.warning(f'No trajectories found in {args.trajectory_folder}, using original dataset')

    logger.info(f'dataset length: {len(dataset)}')
    dataset.cut_long_sequence(args.max_seq_length * 0.96) # leave some buffer for augmenting data
    logger.info(f'dataset length: {len(dataset)}, after cut')


    # Split the dataset into training and validation sets
    # Calculate the validation size (15% of 300 or dataset size, whichever is smaller)
    val_size = min(50, int(0.15 * len(dataset)))
    # Calculate the training size
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = split_dataset_and_keeping_the_oder(dataset, [train_size, val_size])

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Number of CPU cores: {num_cores}")

    batches_per_epoch = rounded_samples_per_epoch // hyperparams.batch_size

    train_loader = CustomDataLoader(train_dataset, batch_size=hyperparams.batch_size, max_batches_in_memory=min(30, batches_per_epoch), 
                                    num_workers=num_cores, collate_fn=dataset.pad_collate)
    val_loader = CacheAllDataLoader(val_dataset, batch_size=int(hyperparams.batch_size * 1.2), collate_fn=dataset.pad_collate)

    # Start the loaders
    train_loader.start(args.seed)


    # Initialize UnifiedTrainingScheduler with immutable config
    scheduler = UnifiedTrainingScheduler(
        optimizer=optimizer,
        run_dir=run_dir,
        config_path=args.scheduler_config,
        previous_state_path=args.scheduler_previous_state
    )
    logger.info(f"Using UnifiedTrainingScheduler")

    # Create a SummaryWriter instance with the unique run directory
    writer = SummaryWriter(run_dir)
    # Add data_sources to TensorBoard
    writer.add_text("Data Sources", ", ".join(data_sources), 0)
    writer.add_text("Git", get_git_hash(), 0)
    writer.add_text("Datetime", datetime.now().strftime("%Y%m%d_%H%M%S"), 0)
    writer.add_text("Env", get_env(), 0)
    writer.add_text("relay_seed", str(relay_seed), 0)

    file_logger_path = os.path.expanduser('~/training.log')
    logger.info(f'file_logger_path {file_logger_path}')
    file_logger = setup_file_logger('batch_logger', file_logger_path)

    logger.info('training starting')
    nvtx.range_push("Main loop")
    start_time = time.time()



    try:        

        # Training loop
        for epoch in range(args.epochs):
            nvtx.range_push(f"Epoch {epoch}")
            
            # Update attempt length from UnifiedTrainingScheduler
            current_attempt_length = scheduler.get_current_attempt_length()
            if hasattr(dataset, 'set_attempt_length'):
                dataset.set_attempt_length(current_attempt_length)
            if current_attempt_length > 0 or epoch == 0:
                logger.info(f"Epoch {epoch}: Attempt length = {current_attempt_length}")
            
            # Log attempt length to TensorBoard at start of epoch
            writer.add_scalar("Training/attempt_length", current_attempt_length, epoch)

            epoch_start_time = time.time()
            train_loss, samples_count, batch_count = train(
                model, train_loader, optimizer, criterion, device, 
                scheduler=scheduler,  # Always pass scheduler for batch-level updates 
                samples_per_epoch=rounded_samples_per_epoch, 
                epoch=epoch, 
                logger=logger,
                accumulation_steps=args.accumulation_steps,
                file_logger=file_logger
            )
            train_end_time = time.time()
            val_loss = validate(model, val_loader, criterion, device, epoch=epoch, logger=logger, file_logger=file_logger)
            val_end_time = time.time()

            nvtx.range_push("post-epoch")

            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024) if device.type == 'cuda' else 0

            logger = setup_logger(overwrite_line=True)
            # Log losses to TensorBoard
            avg_time_per_epoch = (time.time() - start_time) / (epoch + 1)
            remaining_epochs = args.epochs - (epoch + 1)
            eta = timedelta(seconds=int(avg_time_per_epoch * remaining_epochs))
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch}/{args.epochs}({batch_count}b{samples_count}s/e), Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, time:{(train_end_time - epoch_start_time):.2f}+{(val_end_time - train_end_time):.2f}, LR: {current_lr:.2e}, cuda:r{gpu_memory_reserved:.0f}a{gpu_memory_allocated:.0f}MB, ETA: {eta}")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Learning_rate", current_lr, epoch)  # Log the learning rate
            
            try:
                if epoch % 23 == 0:
                    for name, param in model.named_parameters():
                        writer.add_histogram(f"Params/{name}", param.detach(), epoch)
                        if param.grad is not None:
                            writer.add_histogram(f"Grads/{name}", param.grad.detach(), epoch)
            except Exception as e:
                logger.warning(f'Error while logging histograms: {e}')

            best_val_loss = checkpoint_handler.save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, vars(args), start_epoch, args.seed, relay_seed
            )

            if best_val_loss:
                writer.add_scalar("Loss/best_val_loss", best_val_loss, epoch)

            # Step scheduler at epoch level
            assert hasattr(scheduler, 'epoch_step'), "Scheduler must have epoch_step method"
            scheduler.epoch_step()

            # If GPU memory is low, clear the cache
            if is_gpu_memory_low():
                torch.cuda.empty_cache()

            nvtx.range_pop()
            nvtx.range_pop()

            if debug_single_batch:
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Shutting down...")
    finally:
        train_loader.stop()

        print("") # end the last overwrite_line 
        logger = setup_logger(overwrite_line=False)
        total_training_time = timedelta(seconds=int(time.time() - start_time))
        logger.info(f"Training is done({total_training_time}), {hyperparams}")

        if args.log_hyperparameters:
            max_memory_allocated = torch.cuda.max_memory_allocated()
            log_hyperparameters(hyperparams, train_loss, val_loss, max_memory_allocated, writer)
        
        # Close the SummaryWriter
        writer.close()

        nvtx.range_pop()

if __name__ == "__main__":
    main()
