from torch.optim import Adam
import torch


def get_submodule_by_name(model, name):
    # Supports nested names like "encoder.lstm"
    current = model
    for attr in name.split('.'):
        current = getattr(current, attr)
    return current

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
    """
    # Support DataParallel by using .module for parameter lookup
    real_model = model.module if hasattr(model, "module") else model

    opt_name = opt_config['name']
    args = opt_config.get('args', {})
    param_groups = opt_config.get('param_groups', None)

    all_param_ids = set()
    param_to_lr = {}

    if param_groups:
        grouped_params = []
        for group in param_groups:
            param_names = group['params']
            group_options = {k: v for k, v in group.items() if k != 'params'}

            # If learning rate is 0, skip this group
            if 'lr' in group_options and group_options['lr'] == 0:
                continue

            params = []
            for name in param_names:
                # Try submodule lookup
                try:
                    submodule = get_submodule_by_name(real_model, name)
                    for n, p in submodule.named_parameters(recurse=True):
                        params.append(p)
                        param_to_lr[p] = group_options.get('lr', args.get('lr', None))
                        all_param_ids.add(id(p))
                except AttributeError:
                    # Fallback to named parameter
                    p = dict(real_model.named_parameters()).get(name, None)
                    if p is not None:
                        params.append(p)
                        param_to_lr[p] = group_options.get('lr', args.get('lr', None))
                        all_param_ids.add(id(p))
                    else:
                        raise ValueError(f"Parameter or submodule '{name}' not found in model.")
            if params:
                group_entry = {'params': params}
                group_entry.update(group_options)
                grouped_params.append(group_entry)
        if not grouped_params:
            raise ValueError("No parameter groups with nonzero learning rate found.")
        opt_params = grouped_params
    else:
        # Fallback: all parameters, assign global lr
        opt_params = real_model.parameters()
        for n, p in real_model.named_parameters():
            param_to_lr[p] = args.get('lr', None)
            all_param_ids.add(id(p))

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
        raise ValueError(f"Unsupported optimizer: {opt_name}")

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

