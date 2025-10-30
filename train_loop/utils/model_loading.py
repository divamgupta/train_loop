import torch
import io
import hashlib
from .download import download_file
import os
import re
import glob

def get_latest_checkpoint(save_dir):
    """
    Find the latest checkpoint file in the save directory.
    First checks for model_latest.pt, then falls back to model_epoch_*.pt files.
    
    Args:
        save_dir: Directory to search for checkpoints
        
    Returns:
        tuple: (checkpoint_path, epoch_number) or (None, 0) if no checkpoints found
    """
    # First check for model_latest.pt
    latest_path = os.path.join(save_dir, 'model_latest.pt')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location='cpu')
        epoch_num = checkpoint['epoch']
        return latest_path, epoch_num
       
    # Fall back to looking for checkpoint files matching the pattern
    checkpoint_pattern = os.path.join(save_dir, 'model_epoch_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None, 0
    
    # Extract epoch numbers and find the latest
    latest_epoch = 0
    latest_checkpoint = None
    
    for checkpoint_file in checkpoint_files:
        # Extract epoch number from filename
        match = re.search(r'model_epoch_(\d+)\.pt', checkpoint_file)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = checkpoint_file
    
    return latest_checkpoint, latest_epoch


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

