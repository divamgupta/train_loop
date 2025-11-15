import os
import torch
from omegaconf import OmegaConf
from .model_loading import get_latest_checkpoint
from .dynamic_import import build_class


def load_model_from_config_path(config):
    """
    Loads a model from a config object or config path, builds the model using build_class, and loads the latest checkpoint.
    Args:
        config: OmegaConf config object or path to config file (yaml/yml).
    Returns:
        model: The model loaded with the latest checkpoint (if available).
        config: The loaded config object (OmegaConf).
    """

    # If config is a path, load it
    if isinstance(config, str) and not ((config.endswith(".yml") or config.endswith(".yaml"))):
        config = os.path.join(config, "config.yaml")

    if isinstance(config, str):
        config = OmegaConf.load(config)
    # Build model using build_class
    model = build_class(config.model)
    device = config.train.device if hasattr(config.train, 'device') else 'cpu'
    model.to(device)
    # Load latest checkpoint if available
    save_dir = config.train.save_dir
    latest_checkpoint, _ = get_latest_checkpoint(save_dir)
    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"No checkpoint found in {save_dir}. Model initialized with random weights.")
    return model

