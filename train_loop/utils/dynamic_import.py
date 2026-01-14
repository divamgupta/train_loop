import importlib
from .model_loading import load_model_weights, load_weights_pt
import copy

def build_class(config, extra_args={}):
    """
    Dynamically load and instantiate a class from config.
    """
    class_name = config['name']
    module_path, class_name = class_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    args = config.get('args', {})
    ckpt_path = config.get('ckpt_path', None)
    if len(extra_args) > 0:
        args = copy.deepcopy(args)
        for k in extra_args:
            args[k] = extra_args[k]
    m = cls(**args)
    if ckpt_path is not None:
        if "ckpt_ignore_prefix" in config and config['ckpt_ignore_prefix'] is not None:
            ignore_prefix = config['ckpt_ignore_prefix']
            load_weights_pt(m,  load_model_weights(ckpt_path) , ignore_prefix=ignore_prefix)
        else:
            m.load_state_dict(load_model_weights(ckpt_path), strict=True)
    return m

def get_obj(class_name, return_none_if_not_found=False):
    if '.' not in class_name and return_none_if_not_found:
        return None
    elif '.' not in class_name:
        raise ValueError(f"Class name {class_name} is not a valid module path.")
    module_path, class_name = class_name.rsplit('.', 1)
    module = None
    cls = None
    if return_none_if_not_found:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            return None
        try:
            cls = getattr(module, class_name)
        except AttributeError:
            return None
        return cls
    else:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls