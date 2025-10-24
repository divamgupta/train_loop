import importlib
import requests
import os
import hashlib
import getpass
from urllib.parse import urlparse
import torch
import io
import numpy as np
from io import BytesIO
from PIL import Image


# Global variables to store credentials in memory
_cached_username = None
_cached_password = None

def load_model_weights(model_path):
    """Load model weights, trying direct load first, then fallback to 'model_state_dict' key"""
    model_path = download_file(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Try to load directly first
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        return checkpoint


def _get_credentials():
    """
    Get credentials from user input and cache them globally.
    Only prompts if credentials are not already cached.
    """
    global _cached_username, _cached_password
    
    if _cached_username is None or _cached_password is None:
        print("Authentication required:")
        _cached_username = input("Username: ")
        _cached_password = getpass.getpass("Password: ")
    
    return _cached_username, _cached_password

def build_class(config):
    """
    Dynamically load and instantiate a class from config.
    
    Args:
        config: Dict with 'name' (module.ClassName) and optional 'args' (dict of arguments)
    
    Returns:
        Instance of the specified class
    """
    class_name = config['name']
    module_path, class_name = class_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    args = config.get('args', {})
    ckpt_path = config.get('ckpt_path', None)
    
    m = cls(**args)

    if ckpt_path is not None:
        if "ckpt_ignore_prefix" in config and config['ckpt_ignore_prefix'] is not None:
            ignore_prefix = config['ckpt_ignore_prefix']
            load_weights_pt(m,  load_model_weights(ckpt_path) , ignore_prefix=ignore_prefix)

        else:
            m.load_state_dict(load_model_weights(ckpt_path), strict=True)

    return m

def download_file(url):

    if not url.startswith("http"):
        return url 

    download_dir = os.path.expanduser("~/.downloads")
    os.makedirs(download_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Extract filename from URL
    parsed_url = urlparse(url)
    url_filename = os.path.basename(parsed_url.path)
    
    if url_filename:
        # Use original filename with hash prefix
        name, ext = os.path.splitext(url_filename)
        filename = f"{name}_{url_hash}{ext}"
    else:
        # No filename in URL, use just the hash
        filename = url_hash
    
    file_path = os.path.join(download_dir, filename)

    print(file_path)
    
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return file_path
    
    # Try without authentication first
    response = requests.get(url)
    
    # If authentication is required, get credentials and retry
    if response.status_code == 401:
        username, password = _get_credentials()
        response = requests.get(url, auth=(username, password))
    
    response.raise_for_status()  # Raise an exception for bad status codes
    
    with open(file_path, "wb") as f:
        f.write(response.content)
    
    return file_path


def load_weights_pt(model, weights, ignore_prefix=None):
    """
    Load weights into a PyTorch model with strict loading, optionally ignoring keys with specific prefix.
    
    Args:
        model: PyTorch model to load weights into
        weights: Either path to checkpoint file or state_dict
        ignore_prefix: String or list of strings - prefixes to ignore when loading
    """
    
    # Handle weights input
    if isinstance(weights, str):
        checkpoint = torch.load(weights, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = weights
    
    # Get current model state dict
    model_state_dict = model.state_dict()
    
    # Handle ignore_prefix
    if ignore_prefix is not None:
        if isinstance(ignore_prefix, str):
            ignore_prefix = [ignore_prefix]
        
        # Update only keys that don't match any ignore prefix
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
        # Update all matching keys
        model_state_dict.update({k: v for k, v in state_dict.items() if k in model_state_dict})
    
    # Load with strict=True
    model.load_state_dict(model_state_dict, strict=True)
    
    return model


def get_model_weights_hash(model, algorithm='sha256'):
    """
    Compute a hash of the model's weights.
    
    Args:
        model: PyTorch model
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', etc.)
    
    Returns:
        String containing the hexadecimal hash digest
    """
    hasher = hashlib.new(algorithm)
    
    # Get state dict and sort keys for consistent ordering
    state_dict = model.state_dict()
    
    for key in sorted(state_dict.keys()):
        # Convert tensor to bytes and update hash
        tensor = state_dict[key]
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        hasher.update(buffer.getvalue())
    
    return hasher.hexdigest()


def show_tensor(tensor, title=None, save_to=None):
    """
    Visualize numpy or pytorch tensors as images.

    Handles:
    - Automatic format detection (CHW, HWC, or batched)
    - Range normalization (0-1, 0-255, or arbitrary ranges)
    - Grayscale and RGB images
    - Jupyter and terminal environments

    Args:
        tensor: numpy array or torch tensor
        title: optional title for the image
        save_to: optional path to save the image file
    """
    # Convert torch tensor to numpy if needed
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu().numpy()
    
    img = np.array(tensor)
    
    # Handle batched tensors - take first image
    if img.ndim == 4:
        img = img[0]
    
    # Handle different channel layouts
    if img.ndim == 3:
        # If channels first (C, H, W), convert to (H, W, C)
        if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        
        # Squeeze single channel grayscale
        if img.shape[2] == 1:
            img = img.squeeze(2)
    
    # Normalize to 0-255 range
    if img.dtype == np.uint8:
        pass  # Already in correct range
    elif img.max() <= 1.0 and img.min() >= 0.0:
        img = (img * 255).astype(np.uint8)
    elif img.max() <= 1.0 and img.min() >= -1.0:
        img = ((img + 1) * 127.5).astype(np.uint8)
    else:
        # Normalize arbitrary range to 0-255
        img = img - img.min()
        img = (img / (img.max() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Save image if save_to is provided
    if save_to is not None:
        pil_img.save(save_to)
        print(f"Image saved to {save_to}")
        return

    # Check if in Jupyter environment
    try:
        get_ipython()
        in_jupyter = True
    except NameError:
        in_jupyter = False
    
    if in_jupyter:
        from IPython.display import display
        if title:
            print(title)
        display(pil_img)
    else:
        # Use imgcat Python module for terminal
        from imgcat import imgcat
        
        if title:
            print(f"\n{title}")
        
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        imgcat(buf.getvalue())  # Pass bytes instead of BytesIO object
