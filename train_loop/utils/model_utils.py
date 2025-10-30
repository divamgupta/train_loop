import torch.nn as nn
import torch
import hashlib



def params_count_mb(model: nn.Module, depth: int = 1, prefix: str = '') -> dict:
    """
    Returns a breakdown of parameter counts (in Megabytes, assuming FP32) for a model and its submodules.
    
    Args:
        model (nn.Module): The model to analyze.
        depth (int): How deep to go into submodules (1 = top-level only, 2 = children, etc.).
        prefix (str): Used internally for recursion. Should not be set manually.

    Returns:
        dict: {module_name: param_size_MB}
    """
    breakdown = {}
    param_size = lambda p: p.numel()  / (1024 ** 2)  # FP32: 4 bytes per param

    # Calculate total params for this module (including all children)
    total_params = sum(param_size(p) for p in model.parameters())
    breakdown[prefix if prefix else model._get_name()] = total_params

    # Dive into children if depth > 1
    if depth > 1:
        for name, child in model.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            child_breakdown = params_count_mb(child, depth=depth-1, prefix=child_prefix)
            breakdown.update(child_breakdown)

    return breakdown

def print_params_count_mb(model: nn.Module, depth: int = 1):
    """
    Prints the parameter count breakdown in MB for a model up to a certain depth.
    """
    breakdown = params_count_mb(model, depth)
    for mod, size in breakdown.items():
        print(f"  {mod}: {size:.2f} M")

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

def move_to_device(batch, device):
    """
    Recursively moves all tensors in a batch to the specified device.
    Handles dictionaries, lists, and tensors.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        return batch

