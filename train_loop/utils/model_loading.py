import torch
import io
import hashlib
from .download import download_file
import os
import re
import glob
import time

def get_latest_checkpoint(save_dir):
    """
    Find the latest checkpoint file in the save directory.
    First checks for model_latest.pt, then falls back to model_step_*.pt files.
    
    Args:
        save_dir: Directory to search for checkpoints
        
    Returns:
        tuple: (checkpoint_path, step_number) or (None, 0) if no checkpoints found
    """
    # First check for model_latest.pt
    latest_path = os.path.join(save_dir, 'model_latest.pt')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        step_num = checkpoint['n_steps_done']
        return latest_path, step_num
       
    # Fall back to looking for checkpoint files matching the pattern
    checkpoint_pattern = os.path.join(save_dir, 'model_step_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None, 0
    
    # Extract step numbers and find the latest
    latest_step = 0
    latest_checkpoint = None
    
    for checkpoint_file in checkpoint_files:
        # Extract step number from filename
        match = re.search(r'model_step_(\d+)\.pt', checkpoint_file)
        if match:
            step_num = int(match.group(1))
            if step_num > latest_step:
                latest_step = step_num
                latest_checkpoint = checkpoint_file
    
    return latest_checkpoint, latest_step


def load_model_weights(model_path):
    """Load model weights, trying direct load first, then fallback to 'model_state_dict' key"""
    model_path = download_file(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        return checkpoint

def load_weights_pt(model, weights, ignore_prefix=None):
    """
    Load weights into a PyTorch model with strict loading, optionally ignoring keys with specific prefix.
    """
    if isinstance(weights, str):
        checkpoint = torch.load(weights, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = weights
    model_state_dict = model.state_dict()
    if ignore_prefix is not None:
        if isinstance(ignore_prefix, str):
            ignore_prefix = [ignore_prefix]
        for key, value in state_dict.items():
            if key in model_state_dict and not any(key.startswith(prefix) for prefix in ignore_prefix):
                model_state_dict[key] = value
            if key not in model_state_dict and not any(key.startswith(prefix) for prefix in ignore_prefix):
                raise KeyError(f"Key '{key}' not found in model state_dict")
        for key in model_state_dict.keys():
            if not any(key.startswith(prefix) for prefix in ignore_prefix):
                if key not in state_dict:
                    raise KeyError(f"Key '{key}' not found in provided state_dict")
    else:
        model_state_dict.update({k: v for k, v in state_dict.items() if k in model_state_dict})
    model.load_state_dict(model_state_dict, strict=True)
    return model


def resume_from_checkpoint( resume_config, is_master , device , optimizer, model_module, gan_loss_module, disc_optimizer):
    """
    resume_config: An object with attributes:
        - resume (bool): Whether to resume from the latest checkpoint in save_dir.
        - save_dir (str): Directory to look for checkpoints.
        - resume_checkpoint_path (str or None): Specific checkpoint path to resume from.
        - resume_steps_num (int or None): Step number to resume from if specified.
    """
    was_resumed = False
    n_steps_done = -1
    if resume_config.resume:
        # Auto-find latest checkpoint
        latest_checkpoint, latest_n_steps = get_latest_checkpoint( resume_config.save_dir )
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint, map_location=device , weights_only=False)
            # For DataParallel/DDP, load state_dict to .module
            model_module.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if gan_loss_module is not None and 'gan_loss_state_dict' in checkpoint:
                gan_loss_module.load_state_dict(checkpoint['gan_loss_state_dict'])
            if disc_optimizer is not None and 'disc_optimizer_state_dict' in checkpoint:
                disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            n_steps_done = latest_n_steps
            print(f"Auto-resuming from latest checkpoint: {latest_checkpoint} at step {n_steps_done}")
            was_resumed = True

            del checkpoint  # free memory
        else:
            print("No checkpoints found for resuming, starting from scratch")
    elif resume_config.resume_checkpoint_path is not None and (os.path.exists(resume_config.resume_checkpoint_path) or resume_config.resume_checkpoint_path.startswith("http")):
        if resume_config.resume_checkpoint_path.startswith("http"):
            # Download the checkpoint file
            if is_master:
                _ = download_file(resume_config.resume_checkpoint_path)
            else:
                time.sleep(2)
                # Wait for the master to download
                print("Waiting for master to download checkpoint...")
                
            torch.distributed.barrier()
            print("Loading checkpoint from URL...")
            checkpoint_path = download_file(resume_config.resume_checkpoint_path)
            resume_config.resume_checkpoint_path = checkpoint_path
                
        # Manual checkpoint path specified
        checkpoint = torch.load(resume_config.resume_checkpoint_path, map_location=device, weights_only=False)
        model_module.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if gan_loss_module is not None and 'gan_loss_state_dict' in checkpoint:
            gan_loss_module.load_state_dict(checkpoint['gan_loss_state_dict'])
        if disc_optimizer is not None and 'disc_optimizer_state_dict' in checkpoint:
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        if resume_config.resume_steps_num is not None:
            n_steps_done = resume_config.resume_steps_num
        elif 'n_steps_done' in checkpoint:
            n_steps_done = checkpoint['n_steps_done']
        print(f"Resuming from specified checkpoint at step {n_steps_done}")
        was_resumed = True
        
        del checkpoint  # free memory
    return n_steps_done, was_resumed

def save_checkpoint(save_checkpoint_config, is_master  , optimizer, model_module, n_steps_done, loss_history, gan_loss_module, disc_optimizer):
    if is_master:
        if save_checkpoint_config.save_dir is not None and not save_checkpoint_config.no_save_weights:
            checkpoint_path = os.path.join(save_checkpoint_config.save_dir, f'model_step_{n_steps_done}.pt')
            state_dict = model_module.state_dict()
            cur_loss = sum(loss_history['total_loss']) /  len(loss_history['total_loss'])
            to_save = {
                'n_steps_done': n_steps_done,
                'model_state_dict': state_dict,
                'loss': cur_loss,
            }
            if gan_loss_module is not None:
                to_save['gan_loss_state_dict'] = gan_loss_module.state_dict()
            
            
            if save_checkpoint_config.save_separate_stepwise_checkpoints:
                
                if save_checkpoint_config.save_optimizer_in_all_checkpoints:
                    to_save['optimizer_state_dict'] = optimizer.state_dict()
                    if disc_optimizer is not None:
                        to_save['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
                torch.save(to_save, checkpoint_path)

            to_save['optimizer_state_dict'] = optimizer.state_dict()
            if disc_optimizer is not None:
                to_save['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
            torch.save(to_save, os.path.join(save_checkpoint_config.save_dir, 'model_latest.pt'))
            print(f"Saved checkpoint at step {n_steps_done} to {checkpoint_path}")
        elif save_checkpoint_config.save_dir is None:
            pass  # do not save, but do not skip other logic
        else:
            print("Skipping checkpoint save (no_save_weights=True)")
