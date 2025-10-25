import sys 
sys.path.append(".")

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import time
from omegaconf import OmegaConf
import glob
import re
import json
import shutil
import subprocess
from .losses import LossModule 
from .utils.dynamic_import import build_class


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

def evaluate_loss(model, val_dataloader, loss_function, device):
    """
    Evaluate the model on the validation dataset.
    
    Args:
        model: The model to evaluate
        val_dataloader: DataLoader for the validation dataset
        loss_function: Loss function for evaluation
        device: Device to run evaluation on
        
    Returns:
        tuple: (average_loss, loss_components_dict)
    """
    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    dataset = val_dataloader.dataset
    
    with torch.no_grad():
        for batch_inputs_dict in tqdm(val_dataloader, desc="Validation"):
            if batch_inputs_dict is None:
                continue
                
            batch_inputs_dict = move_to_device(batch_inputs_dict, device)
            
            # Get teacher outputs
            if hasattr(dataset , 'post_process_batch'):
                dataset.post_process_batch(batch_inputs_dict)
            
            # Get outputs
            model_outputs = model(batch_inputs_dict)
            
            # Calculate loss
            loss, losses = loss_function(batch_inputs_dict, model_outputs)
            
            total_loss += loss.item()
            
            # Track individual loss components
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v.item()
                
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_components = {k: v / num_batches for k, v in loss_components.items()} if num_batches > 0 else {}
    
    model.train()
    avg_components['total_loss'] = avg_loss
    return avg_loss, avg_components

def save_git_state(save_dir):
    """
    Save the current git state including commit hash and uncommitted changes.
    
    Args:
        save_dir: Directory to save git state information
    """
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get current commit hash
        hash_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        current_hash = hash_result.stdout.strip()
        
        # Get uncommitted changes (staged + unstaged)
        diff_result = subprocess.run(
            ['git', 'diff', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        uncommitted_changes = diff_result.stdout
        
        # Get untracked files
        untracked_result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            capture_output=True,
            text=True,
            check=True
        )
        untracked_files = untracked_result.stdout.strip().split('\n') if untracked_result.stdout.strip() else []
        
        # Save commit hash
        hash_file = os.path.join(save_dir, 'git_commit_hash.txt')
        with open(hash_file, 'w') as f:
            f.write(current_hash + '\n')
        
        # Save uncommitted changes (tracked files) if any exist
        if uncommitted_changes.strip():
            patch_file = os.path.join(save_dir, f'uncommitted_changes_{current_hash[:8]}.patch')
            with open(patch_file, 'w') as f:
                f.write(uncommitted_changes)
            print(f"Saved uncommitted changes to: {patch_file}")
            print(f"  Apply with: git apply {patch_file}")
        
        # Save untracked files separately
        if untracked_files:
            untracked_file_path = os.path.join(save_dir, f'untracked_files_{current_hash[:8]}.txt')
            with open(untracked_file_path, 'w') as f:
                f.write("# Untracked files at training time\n")
                f.write(f"# Commit: {current_hash}\n\n")
                for untracked_file in untracked_files:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"# File: {untracked_file}\n")
                    f.write(f"{'='*80}\n")
                    try:
                        with open(untracked_file, 'r', encoding='utf-8', errors='ignore') as uf:
                            file_content = uf.read()
                        f.write(file_content)
                        if not file_content.endswith('\n'):
                            f.write('\n')
                    except Exception as e:
                        f.write(f"# Error reading file: {e}\n")
            print(f"Saved {len(untracked_files)} untracked file(s) to: {untracked_file_path}")
        
        if not uncommitted_changes.strip() and not untracked_files:
            print("No uncommitted changes or untracked files detected.")
        
        print(f"Git commit hash: {current_hash}")
        
    except subprocess.CalledProcessError:
        print("Warning: Not in a git repository or git command failed. Skipping git state save.")
    except FileNotFoundError:
        print("Warning: Git is not installed. Skipping git state save.")

def train(config):

    if 'sanity' in config and config.sanity:
        config.train.epochs = 3
        config.train.num_per_epoch = 36
        config.train.save_dir = "/tmp/sanity_check"
        config.train.num_eval_every_steps=4

        # if exists delete the dir 
        if os.path.exists(config.train.save_dir):
            shutil.rmtree(config.train.save_dir)

    out_config_path = os.path.join(config.train.save_dir, 'config.yaml')
    if os.path.exists(out_config_path) and (not config.train.get('resume', False)):
        raise FileExistsError(f"Config file {out_config_path} already exists. Please remove it or choose a different save directory.")
    
    # Multi-GPU support
    n_gpus = config.train.get('n_gpus', 1)
    device = config.train.device
    if n_gpus > 1:
        assert torch.cuda.is_available(), "Multi-GPU requested but CUDA is not available."
        device = torch.device("cuda:0")
        print(f"Using {n_gpus} GPUs for training (DataParallel)")
    else:
        device = torch.device(device)

    model = build_class(config.model)
    model.to(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))

    extra_models = {}
    if 'extra_models' in config and config.extra_models is not None:
        for name, model_cfg in config.extra_models.items():
            extra_model = build_class(model_cfg)
            extra_model.to(device)
            if n_gpus > 1:
                extra_model = torch.nn.DataParallel(extra_model, device_ids=list(range(n_gpus)))
            extra_models[name] = extra_model

    gan_loss = None
    if 'gan_loss' in config and config.gan_loss is not None:
        gan_loss = build_class(config.gan_loss)
        gan_loss.to(device)
        if n_gpus > 1:
            gan_loss = torch.nn.DataParallel(gan_loss, device_ids=list(range(n_gpus)))

    dataset = build_class(config.dataset)
    # Setup dataloader
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                           collate_fn=dataset.collate_fn, num_workers=config.train.num_workers, shuffle=True)

    # Initialize validation dataset if specified in config
    val_dataset = None
    val_dataloader = None
    if 'val_dataset' in config:
        val_dataset = build_class(config.val_dataset)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.train.get('val_batch_size', config.train.batch_size),
            collate_fn=val_dataset.collate_fn, 
            num_workers=config.train.get('val_num_workers', config.train.num_workers),
            shuffle=False
        )

    dataset.extra_models = extra_models
    if val_dataset is not None:
        val_dataset.extra_models = extra_models

    # Load checkpoint if specified
    start_epoch = 0
    optimizer = get_opt(model, config.train.optimizer)
    disc_optimizer = None
    if gan_loss is not None:
        disc_optimizer = get_opt(gan_loss, config.train.gan_optimizer)

    if 'name' in config.losses:
        loss_function = build_class(config.losses)
    else:
        loss_function = LossModule(**config.losses)

    # Track best validation loss for saving best model
    best_val_loss = float('inf')

    

    # Handle checkpoint resuming
    if config.train.get('resume', False):
        # Auto-find latest checkpoint
        latest_checkpoint, latest_epoch = get_latest_checkpoint(config.train.save_dir)
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint)
            # For DataParallel, load state_dict to .module
            if n_gpus > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = latest_epoch
            print(f"Auto-resuming from latest checkpoint: {latest_checkpoint} at epoch {start_epoch}")
        else:
            print("No checkpoints found for resuming, starting from scratch")
    elif config.train.get('resume_checkpoint_path') and os.path.exists(config.train.resume_checkpoint_path):
        # Manual checkpoint path specified
        checkpoint = torch.load(config.train.resume_checkpoint_path)
        if n_gpus > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if config.train.get('resume_epoch_num') is not None:
            start_epoch = config.train.resume_epoch_num
        elif 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            
        print(f"Resuming from specified checkpoint at epoch {start_epoch}")
    
    

    def inf_gen():
        while True:
            for x in dataloader:
                yield x
    
    # Calculate iterations per epoch
    if config.train.num_per_epoch <= 0:
        iterations_per_epoch = len(dataset) // config.train.batch_size
        if len(dataset) % config.train.batch_size != 0:
            iterations_per_epoch += 1
    else:
        iterations_per_epoch = config.train.num_per_epoch // config.train.batch_size
        if config.train.num_per_epoch % config.train.batch_size != 0:
            iterations_per_epoch += 1
    
    # Create infinite generator
    data_generator = inf_gen()

    # Make sure save directory exists
    os.makedirs(config.train.save_dir, exist_ok=True)

    # Save git state for reproducibility
    save_git_state(config.train.save_dir)

    # Save the final config to the run directory
    os.makedirs(config.train.save_dir, exist_ok=True)
    
    with open(out_config_path, 'w') as f:
        OmegaConf.save(config, f)
    
    # Put models in training/eval mode
    model.train()
    # If extra models and gan_loss, set train mode

    if gan_loss is not None:
        gan_loss.train()

    # Get evaluation frequency
    eval_frequency = config.train.get('num_eval_every_steps', 1000)
    global_step = 0
    
    # Training loop
    for epoch in range(start_epoch, config.train.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        # Track rolling average of losses over last 30 iterations
        loss_history = {}
        rolling_window = 30

        if "summary_functions" in config:
            for func in config.summary_functions:
                summary_func = build_class(func)
                summary_func.run(model, dataset, config.train.save_dir, epoch)
        
        progress_bar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch+1}/{config.train.epochs} - Avg Loss: 0.000000")
        
        for batch_idx in progress_bar:
            batch_inputs_dict = next(data_generator)

            if batch_inputs_dict is None:
                print("Batch is None, skipping...")
                continue
            
            batch_inputs_dict = move_to_device(batch_inputs_dict, device)

            if hasattr(dataset , 'post_process_batch'):
                dataset.post_process_batch(batch_inputs_dict)
            

            model_outputs = model(batch_inputs_dict)

            if gan_loss is not None:
                # For DataParallel, access .module if needed
                disc_loss_fn = gan_loss
                if n_gpus > 1 and hasattr(gan_loss, 'module'):
                    disc_loss_fn = gan_loss.module
                loss_disc = disc_loss_fn.discriminator_loss(batch_inputs_dict['teacher_audio'], model_outputs['final_audio'])
                disc_optimizer.zero_grad()
                loss_disc.backward(retain_graph=True)
                disc_optimizer.step()

            loss_fn = loss_function
            if n_gpus > 1 and hasattr(loss_function, 'module'):
                loss_fn = loss_function.module
            loss, losses = loss_fn(batch_inputs_dict, model_outputs)

            if gan_loss is not None:
                gen_loss_fn = gan_loss
                if n_gpus > 1 and hasattr(gan_loss, 'module'):
                    gen_loss_fn = gan_loss.module
                gen_loss = gen_loss_fn.generator_loss(batch_inputs_dict['teacher_audio'], model_outputs['final_audio'])
                loss += gen_loss * config.gan_loss.gen_loss_weight
                losses['gen_loss'] = gen_loss
                losses['disc_loss'] = loss_disc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update rolling average for each loss component
            for k, v in losses.items():
                if k not in loss_history:
                    loss_history[k] = []
                loss_history[k].append(v.item())
                if len(loss_history[k]) > rolling_window:
                    loss_history[k].pop(0)
            
            # Log training loss to jsonl file every rolling_window steps
            if global_step % rolling_window == 0:
                train_loss_components = {k: sum(v)/len(v) for k, v in loss_history.items()}
                train_loss_components['step'] = global_step
                train_loss_components['epoch'] = epoch
                train_loss_components['total_loss'] = loss.item()
                
                train_loss_file = os.path.join(config.train.save_dir, 'train_loss.jsonl')
                with open(train_loss_file, 'a') as f:
                    f.write(json.dumps(train_loss_components) + '\n')

            # if at epoch 0 and iter 400, if losss is high then reset the training! 
            loss_check_iter = config.train.get('loss_check_iter', 400)
            if epoch == 0 and batch_idx == loss_check_iter: #TODO make it configurable
                if 'audio_l1' in loss_history:
                    l = sum(loss_history['audio_l1']) / len(loss_history['audio_l1'])
                    if l > 0.03:
                        print("Loss is too high, restarting training.")
                        return train(config) #todo make this better
            
            # Run validation if needed
            if val_dataloader is not None and eval_frequency > 0 and global_step % eval_frequency == 0:
                val_loss_fn = loss_function
                if n_gpus > 1 and hasattr(loss_function, 'module'):
                    val_loss_fn = loss_function.module
                val_loss, val_loss_components = evaluate_loss(model, val_dataloader, val_loss_fn, device)
                
                # Append validation loss to jsonl file
                val_loss_components['step'] = global_step
                val_loss_components['epoch'] = epoch
                val_loss_file = os.path.join(config.train.save_dir, 'val_loss.jsonl')
                with open(val_loss_file, 'a') as f:
                    f.write(json.dumps(val_loss_components) + '\n')
                
                print(f"Validation Loss at step {global_step}: {val_loss:.6f}")
                print(f"Validation Loss Components: {val_loss_components}")
            
            current_avg_loss = epoch_loss / (batch_idx + 1)
            # Calculate rolling average for each loss component
            loss_str = " | ".join([f"{k}: {sum(history)/len(history):.4f}" for k, history in loss_history.items()])
            progress_bar.set_description(f"Epoch {epoch+1}/{config.train.epochs} - Avg Loss: {current_avg_loss:.6f} | {loss_str}")
            
            
        # Calculate average loss
        avg_loss = epoch_loss / iterations_per_epoch
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{config.train.epochs} completed in {elapsed_time:.2f}s - Avg Loss: {avg_loss:.6f}")
        # Save checkpoint
        if not config.train.get('no_save_weights', False):
            checkpoint_path = os.path.join(config.train.save_dir, f'model_epoch_{epoch+1}.pt')
            state_dict = model.module.state_dict() if n_gpus > 1 else model.state_dict()
            if not config.train.get('no_save_epoch_wise_weights', False):
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': state_dict,
                    'loss': avg_loss,
                }, checkpoint_path)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(config.train.save_dir, 'model_latest.pt'))
        else:
            print("Skipping checkpoint save (no_save_weights=True)")
    
    print("Training completed!")
    print("Saved in " , config.train.save_dir)
    return model


def train_cli():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('overrides', nargs='*',
                        help='Override config values using dotpath notation (e.g., train.lr=0.001, model.hidden_size=256)')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    # Apply config overrides using OmegaConf CLI
    if args.overrides:
        override_config = OmegaConf.from_dotlist(args.overrides)
        
        # Validate that all override keys exist in the original config
        def validate_keys(override_cfg, original_cfg, path=""):
            for key in override_cfg:
                current_path = f"{path}.{key}" if path else key
                
                if key not in original_cfg:
                    raise KeyError(f"Override key '{current_path}' does not exist in the original config")
                if OmegaConf.is_config(override_cfg[key]) and OmegaConf.is_config(original_cfg[key]):
                    validate_keys(override_cfg[key], original_cfg[key], current_path)

        if override_config.get("validate_override_keys", True):
            validate_keys(override_config, config)
        
        config = OmegaConf.merge(config, override_config)
        print(f"Applied overrides: {args.overrides}")
    
    print("Loaded config:")
    print(OmegaConf.to_yaml(config))
    train(config)

if __name__ == '__main__':
    train_cli()