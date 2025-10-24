import torch
import io
import hashlib
from .download import download_file

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

def get_model_weights_hash(model, algorithm='sha256'):
    """
    Compute a hash of the model's weights.
    """
    hasher = hashlib.new(algorithm)
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        hasher.update(buffer.getvalue())
    return hasher.hexdigest()
