import importlib
from .model_loading import load_model_weights, load_weights_pt

def build_class(config):
    """
    Dynamically load and instantiate a class from config.
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
