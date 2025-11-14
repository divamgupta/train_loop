from torch.optim import Adam
import torch
from .utils.dynamic_import import get_obj

def get_submodule_by_name(model, name):
    # Supports nested names like "encoder.lstm"
    current = model
    for attr in name.split('.'):
        current = getattr(current, attr)
    return current


def get_params_by_name(model, name):
    
    params_dict = dict(model.named_parameters())
    if name in params_dict:
        return [params_dict[name]]
    else:
        submodule = get_submodule_by_name(model, name)
        return [p for n, p in submodule.named_parameters(recurse=True)]

def get_all_param_names(model , names):
    params = []
    for name in names:
        params.extend(get_params_by_name(model, name))
    return params


class MultiOptimizer:
    def __init__(self, **optimizers):
        self.optimizers = optimizers
        self.type = 'multi'
    
    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()
    
    def step(self):
        for opt in self.optimizers.values():
            opt.step()
    
    def state_dict(self):
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}
    
    def load_state_dict(self, state_dicts):
        for name, state in state_dicts.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)


def get_opt(model, opt_config, print_summary=True):
    """
    Returns an optimizer based on the configuration, and optionally prints a summary.

    Args:
        model: Model to optimize (supports DataParallel)
        opt_config: Dict with:
            - 'name' (optimizer type)
            - 'args' (dict of global arguments)
            - (optional) 'param_groups' (list of dicts, each with 'params' and options like 'lr')
              - 'params' can be a list of parameter names or submodule names.
        print_summary: If True, prints summary of assigned learning rates and unoptimized params.

    Returns:
        Optimizer instance

    Example opt_config:
    optimizer:
        name: adam
        args:
            lr: 0.001
        param_groups:
            - params: ["text_encoder" ]
                lr: 0.001
            - params: ["decoder"]
                lr: 0

    or with multi-optimizer:
    optimizer:
        name: multi
        optimizers:
            adam_opt:
                name: adam
                args:
                    lr: 0.001
                param_groups:
                    - params: ["text_encoder"]
                      lr: 0.001
            sgd_opt:
                name: sgd
                args:
                    lr: 0.01
                param_groups:
                    - params: ["decoder"]
                      lr: 0.01

    """
    # Support DataParallel by using .module for parameter lookup
    real_model = model.module if hasattr(model, "module") else model

    opt_name = opt_config['name']
    args = opt_config.get('args', {})
    param_groups = opt_config.get('param_groups', None)

    if opt_name == "multi":
        all_optimizers = {
            name: get_opt(model, sub_opt_config, print_summary)
            for name, sub_opt_config in opt_config['optimizers'].items()
        }
        return MultiOptimizer(**all_optimizers)

    if param_groups:
        all_grouped_params = []
        for group in param_groups:
            param_names = group['params']
            group_options = {k: v for k, v in group.items() if k != 'params'} # Stuff like 'lr' etc for each group

            # If learning rate is 0, skip this group
            if 'lr' in group_options and group_options['lr'] == 0:
                continue

            params = get_all_param_names(model, param_names)

            if not params or len(params) == 0:
                raise ValueError(f"No parameters found for names: {param_names}")

            group_entry = {'params': params}
            group_entry.update(group_options) # Add other options like 'lr', along with params
            all_grouped_params.append(group_entry)
            
        if not all_grouped_params:
            raise ValueError("No parameter groups with nonzero learning rate found.")
        opt_params = all_grouped_params
    else:
        # If no param groups, optimize all parameters with same global lr
        opt_params = real_model.parameters()
       
    # Build optimizer
    if opt_name == "adam":
        opt = Adam(opt_params, **args)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(opt_params, **args)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(opt_params, **args)
    elif opt_name == "rmsprop":
        opt = torch.optim.RMSprop(opt_params, **args)
    else:
        opt_class = get_obj(opt_name)
        opt = opt_class(opt_params, **args)

    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

    if print_summary:
        print("Optimizer parameter summary:")
        param_to_name = {p: n for n, p in real_model.named_parameters()}
        assigned = set()
        for i, group in enumerate(opt.param_groups):
            lr = group.get('lr', args.get('lr', None))
            for p in group['params']:
                name = param_to_name.get(p, "<unnamed>")
                print(f"  [{i}] {name}: lr={lr}")
                assigned.add(id(p))
        # Find unassigned parameters
        print("Parameters NOT in any optimizer group:")
        for n, p in real_model.named_parameters():
            if id(p) not in assigned:
                print(f"  {n}")
    return opt

