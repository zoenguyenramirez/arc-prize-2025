import torch
import math
import logging

from pathlib import Path

from torch.nn.parallel import DataParallel

from src.model import Transformer
from src.utils.logger_helper import setup_logger
from src.token import VOCAB_SIZE, SpecialToken

logger = setup_logger(overwrite_line=False)

latest_checkpoint_saving_frequency = 10
periodic_checkpoint_saving_frequency = 100
class CheckpointHandler:

    def __init__(self, save_dir, model_name="Transformer", minimize_checkpoints=False):
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.minimize_checkpoints = minimize_checkpoints

        self.update_save_dir(save_dir)

    def update_save_dir(self, save_dir):
        self.save_dir = Path(save_dir)
        self.latest_path = self.save_dir / f"{self.model_name}_latest.pt"
        self.save_dir = Path(save_dir)

    def is_quadratic_checkpoint(self, epoch):        
        if epoch < 50 and epoch > 0:
            return False
        # Solve quadratic equation: x^2 + x - (2 * (epoch - initial) / 100) = 0
        a = 1
        b = 1
        c = -2 * (epoch - 50) / 100
        
        # Quadratic formula: (-b + sqrt(b^2 - 4ac)) / (2a)
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return True
        
        x = (-b + math.sqrt(discriminant)) / (2*a)
        
        # Check if x is very close to an integer
        return abs(x - round(x)) < 1e-6
    
    @staticmethod
    def validate_total_epoch(total_epoch: int):
        assert total_epoch % latest_checkpoint_saving_frequency == 0        
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, args, start_epoch, seed, relay_seed):
        checkpoint = {
            'epoch': start_epoch + epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'hyperparameters': args, # the key name was wrong, too late to fix
            'best_val_loss': self.best_val_loss,
            'epoch_in_session': epoch,
            'seed': seed,
            'relay_seed': relay_seed
        }

        new_best_validation_loss = False

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss            
            checkpoint['best_val_loss'] = val_loss
            new_best_validation_loss = True

        if (epoch + 1) % latest_checkpoint_saving_frequency == 0:
            # Save the latest checkpoint
            torch.save(checkpoint, self.latest_path)

        if not self.minimize_checkpoints:
            # Save periodic checkpoint
            if (epoch + 1) % periodic_checkpoint_saving_frequency == 0:
                periodic_path = self.save_dir / f"{self.model_name}_epoch_{epoch:04d}.pt"
                torch.save(checkpoint, periodic_path)

        # Save the best checkpoint
        if new_best_validation_loss and start_epoch + epoch > 50:
            best_path = self.save_dir / f"{self.model_name}_best_{start_epoch + epoch}.pt"
            logger.info(f'best_saved @: {best_path}')
            torch.save(checkpoint, best_path)

            return self.best_val_loss

        return None
    
    @staticmethod
    def load_and_fixup_checkpoint(path, device, *, adjust_max_length = 0):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        return checkpoint


    @staticmethod
    def load_checkpoint(path, model, *, device, optimizer=None, initial_lr=None, adjust_max_length = 0):
        checkpoint = CheckpointHandler.load_and_fixup_checkpoint(path, device, adjust_max_length = adjust_max_length)
        
        if isinstance(model, DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        if optimizer:
            for param_group in checkpoint['optimizer_state_dict']['param_groups']:
                param_group['initial_lr'] = initial_lr
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint, checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
    
    @staticmethod
    def load_checkpoint_in_production(checkpoint_path, device, *, adjust_max_length=0):
        checkpoint = CheckpointHandler.load_and_fixup_checkpoint(checkpoint_path, adjust_max_length=adjust_max_length, device = device)

        args = checkpoint['hyperparameters']
        model_state = checkpoint['model_state_dict']
        
        # Check for MOE configuration in checkpoint
        num_experts = args.get('num_experts', 8)
        num_experts_per_tok = args.get('num_experts_per_tok', 2)
        moe_layers = args.get('moe_layers', None)
        
        # Auto-detect number of experts from router weights if not specified
        if 'num_experts' not in args:
            for key in model_state.keys():
                if 'router.gate.weight' in key:
                    num_experts = model_state[key].shape[0]
                    logging.info(f"Auto-detected num_experts={num_experts} from checkpoint")
                    break
        
        # Auto-detect MOE layers if not specified
        if moe_layers is None:
            detected_moe_layers = []
            for key in model_state.keys():
                if 'router.gate.weight' in key:
                    # Extract layer number from key like "layers.2.feed_forward.router.gate.weight"
                    parts = key.split('.')
                    if len(parts) > 1 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        if layer_idx not in detected_moe_layers:
                            detected_moe_layers.append(layer_idx)
            if detected_moe_layers:
                moe_layers = sorted(detected_moe_layers)
                logging.info(f"Auto-detected moe_layers={moe_layers} from checkpoint")
        
        # Auto-detect num_kv_heads from checkpoint if present
        num_kv_heads = args.get('num_kv_heads', 1)  # Default to 1 if not in checkpoint
        
        model = Transformer(
            VOCAB_SIZE, 
            args['embed_size'], 
            args['num_layers'], 
            args['heads'], 
            num_kv_heads=num_kv_heads,
            max_length=adjust_max_length if adjust_max_length > 0 else args['max_seq_length']
        ).to(device)

        # Load state dict with strict=False to handle MOE checkpoints on non-MOE models
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if unexpected_keys:
            moe_keys = [k for k in unexpected_keys if 'expert' in k or 'router' in k or 'gate' in k]
            if moe_keys:
                print(f"Note: Loading MOE checkpoint into non-MOE model. Ignoring {len(moe_keys)} MOE-related keys.")
                logging.info(f"Loading MOE checkpoint into non-MOE model. Ignoring {len(moe_keys)} MOE-related keys.")
            non_moe_unexpected = [k for k in unexpected_keys if k not in moe_keys]
            if non_moe_unexpected:
                print(f"Warning: Unexpected keys (non-MOE): {non_moe_unexpected[:5]}...")
                logging.warning(f"Unexpected keys (non-MOE): {non_moe_unexpected[:5]}...")
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
            logging.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")

        checkpoint_info = {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss'],
            'val_loss': checkpoint['val_loss']
        }
        logging.info('The checkpoint was saved at epoch %d, train_loss: %f, val_loss: %f, args: %s', 
                    checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss'], args)
        return model, adjust_max_length or args['max_seq_length'], args, checkpoint_info
    

    @staticmethod
    def restore_model_state(model, checkpoint_path, device, *, adjust_max_length=0):
        checkpoint = CheckpointHandler.load_and_fixup_checkpoint(checkpoint_path, adjust_max_length=adjust_max_length, device = device)
        model.load_state_dict(checkpoint['model_state_dict'])

