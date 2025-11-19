import sys 
sys.path.append(".")

import os
import torch

# Add these lines to limit threading
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from omegaconf import OmegaConf
import json
import shutil
from .losses import LossModule 
from .utils.dynamic_import import build_class
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils.git import save_git_state
from .optimizer import get_opt
from .utils.model_loading import get_latest_checkpoint
from .utils.model_utils import move_to_device
from .evaluate_module import evaluate_loss
from .utils.download import download_file
from contextlib import nullcontext



def get_compiled_model_with_loss(model, loss_function, device):
    """
    Wraps the model and loss function into a single callable for compilation.
    """
    class ModelWithLoss(torch.nn.Module):
        def __init__(self, model, loss_function):
            super(ModelWithLoss, self).__init__()
            self.model = model
            self.loss_function = loss_function
        
        def forward(self, batch_inputs):
            model_outputs = self.model(batch_inputs)
            loss, losses = self.loss_function(batch_inputs, model_outputs)
            return  loss , losses

    model_with_loss = ModelWithLoss(model, loss_function)
    model_with_loss.to(device)
    compiled_model_with_loss = torch.compile(model_with_loss, dynamic=False)
    return compiled_model_with_loss


def train(config):

    if config.train.get('pt_single_threaded', True):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    if 'resume_epoch_num' in config.train or 'epochs' in config.train or 'no_save_epoch_wise_weights' in config.train or 'num_per_epoch' in config.train:
        raise NotImplementedError("Epoch-based training is not supported in this training loop. Please use step-based training with 'n_total_steps' or 'n_total_samples'.")

    # DDP support
    use_ddp = config.train.get('use_ddp', False)
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if use_ddp else 0
    world_size = int(os.environ.get('WORLD_SIZE', 1)) if use_ddp else 1
    n_gpus = config.train.get('n_gpus', 1)

    gradient_accum_steps = config.train.get('gradient_accum_steps', 1)

    effective_batch_size = config.train.batch_size * n_gpus * gradient_accum_steps

    find_unused_parameters = config.train.get('find_unused_parameters', True)  

    if use_ddp:
        assert world_size == n_gpus

    is_master = True
    if use_ddp and local_rank != 0:
        is_master = False

    if is_master:
        print("Training Configuration:")
        print(OmegaConf.to_yaml(config))

    if 'sanity' in config and config.sanity:
        config.train.n_total_steps = 18
        config.train.save_dir = "/tmp/sanity_check"
        config.train.num_eval_every_steps=4
        config.train.num_validation_steps = 10
        config.train.checkpoint_save_frequency = 6 
        config.train.summary_frequency = 6

        # if exists delete the dir 
        if is_master:
            if os.path.exists(config.train.save_dir):
                shutil.rmtree(config.train.save_dir)

        if use_ddp:
            time.sleep(2)  # Ensure all processes wait for dir deletion


    n_total_steps = config.train.get('n_total_steps', -1)
    n_total_samples = config.train.get('n_total_samples', -1)
    n_steps_done = 0

    if n_total_steps < 0 and n_total_samples < 0:
        raise ValueError("Either 'n_total_steps' or 'n_total_samples' must be specified and greater than 0.")

    if n_total_steps < 0:
        n_total_steps = n_total_samples // effective_batch_size


    is_compile_model = config.train.get('compile_model', False)
    is_compile_model_with_loss = config.train.get('is_compile_model_with_loss', False)

    grad_clip_value = config.train.get('grad_clip_value', 0)

    if is_compile_model_with_loss:
        assert is_compile_model, "If compiling model with loss, 'compile_model' must be True."


    out_config_path = os.path.join(config.train.save_dir, 'config.yaml')
    if os.path.exists(out_config_path) and (not config.train.get('resume', False)):
        raise FileExistsError(f"Config file {out_config_path} already exists. Please remove it or choose a different save directory.")
    
    use_tqdm = config.train.get('use_tqdm', False )
    if use_ddp:
        use_tqdm = False

    if use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Using DistributedDataParallel on rank {local_rank} of {world_size}")
    else:
        device = config.train.device
        if n_gpus > 1:
            assert torch.cuda.is_available(), "Multi-GPU requested but CUDA is not available."
            device = torch.device("cuda:0")
            print(f"Using {n_gpus} GPUs for training (DataParallel)")
        else:
            device = torch.device(device)

    is_distributed = use_ddp or n_gpus > 1

    if config.train.get("use_bfloat16_autocast", False):
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()


    if config.train.get('create_model_meta_init', False):
        with torch.device("meta"):
            model = build_class(config.model)
        model.to_empty(device=device)
    else:
        model = build_class(config.model)
        model.to(device)

    if hasattr(model, 'init_weights'):
        model.init_weights()

    if 'name' in config.losses:
        loss_function = build_class(config.losses)
    else:
        loss_function = LossModule(**config.losses)

    if 'metrics' in config:
        if 'name' in config.metrics:
            metrics_module = build_class(config.metrics)
        else:
            metrics_module = LossModule(**config.metrics)
    else:
        metrics_module = None

    model_with_loss = None
    if is_compile_model:
        if is_compile_model_with_loss:
            model_with_loss = get_compiled_model_with_loss(model, loss_function, device)
        else:
            model = torch.compile(model, dynamic=False) 
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=find_unused_parameters)
        if model_with_loss is not None:
            model_with_loss = DDP(model_with_loss, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
    elif n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
        if model_with_loss is not None:
            model_with_loss = torch.nn.DataParallel(model_with_loss, device_ids=list(range(n_gpus)))

    model_module = model.module if is_distributed else model

    if is_master:
        print("Model Summary:")
        total_params = sum(p.numel() for p in model_module.parameters())
        trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params/(1024**2):.2f} M")
        print(f"Trainable parameters: {trainable_params/(1024**2):.2f} M")

    extra_models = {}
    if 'extra_models' in config and config.extra_models is not None:
        for name, model_cfg in config.extra_models.items():
            extra_model = build_class(model_cfg)
            extra_model.to(device)
            if use_ddp:
                pass
                # extra_model = DDP(extra_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
            elif n_gpus > 1:
                extra_model = torch.nn.DataParallel(extra_model, device_ids=list(range(n_gpus)))

            for param in extra_model.parameters():
                param.requires_grad = False

            extra_models[name] = extra_model

    gan_loss = None
    if 'gan_loss' in config and config.gan_loss is not None:
        gan_loss = build_class(config.gan_loss)
        gan_loss.to(device)
        if use_ddp:
            gan_loss = DDP(gan_loss, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
        elif n_gpus > 1:
            gan_loss = torch.nn.DataParallel(gan_loss, device_ids=list(range(n_gpus)))

    dataset = build_class(config.dataset)
    # Setup dataloader
    if use_ddp:
        train_sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                               collate_fn=dataset.collate_fn, num_workers=config.train.num_workers, shuffle=False, sampler=train_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                               collate_fn=dataset.collate_fn, num_workers=config.train.num_workers, shuffle=True)

    # Initialize validation dataset if specified in config
    val_dataset = None
    val_dataloader = None
    if is_master:
        if 'val_dataset' in config:
            val_dataset = build_class(config.val_dataset)
            # since validation happens on master, we do not use DistributedSampler
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
    if config.train.get('get_optimizer_fn_name', None) is not None:
        get_opt_fn_name = config.train.get('get_optimizer_fn_name')
        assert get_opt_fn_name.startswith("__model__."), "For now get_optimizer_fn_name must start with '__model__.'"
        get_opt_fn_name = get_opt_fn_name.replace("__model__.", "")
        if hasattr(model_module, get_opt_fn_name):
            get_opt_model_fn = getattr(model_module, get_opt_fn_name)
            optimizer = get_opt_model_fn()
            print("Loaded optimizer from model function:", get_opt_fn_name)
        else:
            raise ValueError(f"Model does not have method {get_opt_fn_name} to get optimizer.")
    else:
        optimizer = get_opt(model_module, config.train.optimizer, print_summary=is_master)
   
    disc_optimizer = None
    if gan_loss is not None:
        disc_optimizer = get_opt(gan_loss, config.train.gan_optimizer, print_summary=is_master)
    
    # Handle checkpoint resuming
    if config.train.get('resume', False):
        # Auto-find latest checkpoint
        latest_checkpoint, latest_n_steps = get_latest_checkpoint(config.train.save_dir)
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint, map_location=device , weights_only=False)
            # For DataParallel/DDP, load state_dict to .module
            model_module.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            n_steps_done = latest_n_steps
            print(f"Auto-resuming from latest checkpoint: {latest_checkpoint} at step {n_steps_done}")

            del checkpoint  # free memory
        else:
            print("No checkpoints found for resuming, starting from scratch")
    elif config.train.get('resume_checkpoint_path') and (os.path.exists(config.train.resume_checkpoint_path) or config.train.resume_checkpoint_path.startswith("http")):
        if config.train.resume_checkpoint_path.startswith("http"):
            # Download the checkpoint file
            if is_master:
                _ = download_file(config.train.resume_checkpoint_path)
            else:
                time.sleep(2)
                # Wait for the master to download
                print("Waiting for master to download checkpoint...")
                
            torch.distributed.barrier()
            print("Loading checkpoint from URL...")
            checkpoint_path = download_file(config.train.resume_checkpoint_path)
            config.train.resume_checkpoint_path = checkpoint_path
                
        # Manual checkpoint path specified
        checkpoint = torch.load(config.train.resume_checkpoint_path, map_location=device, weights_only=False)
        model_module.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if config.train.get('resume_steps_num') is not None:
            n_steps_done = config.train.resume_steps_num
        elif 'n_steps_done' in checkpoint:
            n_steps_done = checkpoint['n_steps_done']
        print(f"Resuming from specified checkpoint at step {n_steps_done}")
        
        del checkpoint  # free memory



    def inf_gen():
        while True:
            for x in dataloader:
                yield x
    
    # Create infinite generator
    data_generator = inf_gen()

    lr_scheduler = None
    if 'lr_scheduler' in config.train:
        lr_scheduler = build_class(config.train.lr_scheduler, extra_args=dict(optimizer=optimizer ))


    # Make sure save directory exists
    if is_master: # not use_ddp or local_rank == 0:
        os.makedirs(config.train.save_dir, exist_ok=True)
        # Save git state for reproducibility
        save_git_state(config.train.save_dir)
        # Save the final config to the run directory
        out_config_path = os.path.join(config.train.save_dir, 'config.yaml')
        with open(out_config_path, 'w') as f:
            OmegaConf.save(config, f)

    # Put models in training/eval mode
    model.train()

    # If extra models and gan_loss, set train mode
    if gan_loss is not None:
        gan_loss.train()
    

    if is_distributed:
        torch.distributed.barrier()

    summary_frequency = config.train.get('summary_frequency', 1000)
    eval_frequency = config.train.get('num_eval_every_steps', 1000)

    if use_tqdm:
        progress_bar = tqdm(range(n_steps_done, n_total_steps), desc=f"Step {n_steps_done}/{n_total_steps} - Avg Loss: 0.000000")
    else:
        progress_bar = range(n_steps_done, n_total_steps)

    iter_times = []
    loss_history = {}
    rolling_window = 30

    # Training loop
    for _ in progress_bar:
        # if use_ddp:
        #     dataloader.sampler.set_e_poch(e_poch)        
        # Track rolling average of losses over last 30 iterations
        
        

        if summary_frequency > 0 and  n_steps_done % summary_frequency == 0:
            if is_distributed:
                torch.distributed.barrier()
            if is_master:
                if "summary_functions" in config and config.summary_functions is not None:
                    for func in config.summary_functions:
                        summary_func = build_class(func)
                        with autocast_ctx:
                            summary_func.run(model_module, dataset, config.train.save_dir, n_steps_done)
        

        iter_start_time = time.time()

        torch.cuda.synchronize()

        for mini_i in range(gradient_accum_steps):
            batch_inputs_dict = next(data_generator)

            if batch_inputs_dict is None:
                print("Batch is None, skipping...")

                if use_ddp:
                    raise ValueError("Dataloader returned None will make it go out of sync.")

                continue
            
            batch_inputs_dict = move_to_device(batch_inputs_dict, device)

            if hasattr(dataset , 'post_process_batch'):
                dataset.post_process_batch(batch_inputs_dict)
            
            with autocast_ctx:

                losses = {}
                model_outputs = None
                if model_with_loss is not None:
                    loss, losses = model_with_loss(batch_inputs_dict)
                else:
                    model_outputs = model(batch_inputs_dict)

                if gan_loss is not None:
                    assert gradient_accum_steps == 1, "GAN loss with gradient accumulation is not supported yet."
                    # For DataParallel, access .module if needed
                    disc_loss_fn = gan_loss
                    if is_distributed and hasattr(gan_loss, 'module'):
                        disc_loss_fn = gan_loss.module
                    assert model_outputs is not None, "Model outputs should not be None when using GAN loss."
                    loss_disc = disc_loss_fn.discriminator_loss(batch_inputs_dict['teacher_audio'], model_outputs['final_audio'])
                    disc_optimizer.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    disc_optimizer.step()

                if model_with_loss is None:
                    assert model_outputs is not None, "Model outputs should not be None when using loss function."
                    loss_fn = loss_function
                    if is_distributed and hasattr(loss_function, 'module'):
                        loss_fn = loss_function.module
                    loss, losses = loss_fn(batch_inputs_dict, model_outputs)

                # --- Compute metrics ---
                metrics_result = None
                if metrics_module is not None:
                    assert model_outputs is not None, "Model outputs should not be None when computing metrics."
                    metrics_fn = metrics_module
                    if is_distributed and hasattr(metrics_module, 'module'):
                        metrics_fn = metrics_module.module
                    _ , metrics_result = metrics_fn(batch_inputs_dict, model_outputs)

                if gan_loss is not None:
                    assert gradient_accum_steps == 1, "GAN loss with gradient accumulation is not supported yet."
                    gen_loss_fn = gan_loss
                    if is_distributed and hasattr(gan_loss, 'module'):
                        gen_loss_fn = gan_loss.module
                    gen_loss = gen_loss_fn.generator_loss(batch_inputs_dict['teacher_audio'], model_outputs['final_audio'])
                    loss += gen_loss * config.gan_loss.gen_loss_weight
                    losses['gen_loss'] = gen_loss
                    losses['disc_loss'] = loss_disc

            
            (loss / gradient_accum_steps).backward()

        grad_clip_enabled = grad_clip_value > 0.0
        if grad_clip_enabled:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            grad_norm = grad_norm_tensor  # GPU tensor -> CPU float (note: cpu-gpu sync point)

        if lr_scheduler is not None:
            lr_scheduler.step(n_steps_done)

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        
        n_steps_done += 1
        
        # Update rolling average for each loss component
        losses_plus_metrics = losses.copy()
        losses_plus_metrics.update(metrics_result or {})
        losses_plus_metrics['total_loss'] = loss 

        if grad_clip_enabled:
            losses_plus_metrics['grad_norm'] = grad_norm

        for k, v in losses_plus_metrics.items():
            if k not in loss_history:
                loss_history[k] = []
            loss_history[k].append(v.item())
            if len(loss_history[k]) > rolling_window:
                loss_history[k].pop(0)
        
        # Log training loss to jsonl file every rolling_window steps
        if n_steps_done % rolling_window == 0:
            train_loss_components = {k: sum(v)/len(v) for k, v in loss_history.items()}
            train_loss_components['step'] = n_steps_done
            
            train_loss_file = os.path.join(config.train.save_dir, 'train_loss.jsonl')
            with open(train_loss_file, 'a') as f:
                f.write(json.dumps(train_loss_components) + '\n')
        
        # Run validation if needed
        if eval_frequency > 0 and n_steps_done % eval_frequency == 0:
            if is_distributed:
                torch.distributed.barrier()
            
            if val_dataloader is not None:
                with autocast_ctx:
                    if is_master:
                        # time.sleep(10)
                        
                        val_loss_fn = loss_function
                        if is_distributed  and hasattr(loss_function, 'module'):
                            val_loss_fn = loss_function.module
                        val_loss, val_loss_components = evaluate_loss(model_module, val_dataloader, val_loss_fn, device, num_val_steps=config.train.get('num_validation_steps', -1), use_tqdm=config.train.get('validation_use_tqdm', False))
                        # get metrics also 
                        if metrics_module is not None:
                            metrics_fn = metrics_module
                            if is_distributed and hasattr(metrics_module, 'module'):
                                metrics_fn = metrics_module.module
                            _ , val_metrics = evaluate_loss(model_module, val_dataloader, metrics_fn, device, num_val_steps=config.train.get('num_validation_steps', -1), use_tqdm=config.train.get('validation_use_tqdm', False))
                            val_loss_components.update(val_metrics)
                        val_loss_components['total_loss'] = val_loss
                        
                        # Append validation loss to jsonl file
                        val_loss_components['step'] = n_steps_done
                        val_loss_file = os.path.join(config.train.save_dir, 'val_loss.jsonl')
                        with open(val_loss_file, 'a') as f:
                            f.write(json.dumps(val_loss_components) + '\n')
                        
                        print(f"Validation Loss at step {n_steps_done}: {val_loss:.6f}")
                        print(f"Validation Loss Components: {val_loss_components}")

            if is_distributed:
                torch.distributed.barrier()
        
        # Calculate rolling average for each loss component
        loss_str = " | ".join([f"{k}: {sum(history)/len(history):.4f}" for k, history in loss_history.items()])
        if use_tqdm:
            progress_bar.set_description(f"Step {n_steps_done}/{n_total_steps} -  {loss_str}")
        else:
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            iter_times.append(iter_time)
            avg_iter_time = sum(iter_times) / len(iter_times)
            remaining_iters = n_total_steps - n_steps_done
            eta = remaining_iters * avg_iter_time
            print(f"Rank {local_rank} Step {n_steps_done}/{n_total_steps} - {loss_str} | Iter time: {iter_time:.2f}s | ETA: {eta:.2f}s")
        

        # Save checkpoint
        if n_steps_done % config.train.get('checkpoint_save_frequency', 1000) == 0 or n_steps_done >= n_total_steps:
            if is_master:
                if not config.train.get('no_save_weights', False):
                    checkpoint_path = os.path.join(config.train.save_dir, f'model_step_{n_steps_done}.pt')
                    state_dict = model_module.state_dict()
                    cur_loss = sum(loss_history['total_loss']) /  len(loss_history['total_loss'])
                    if config.train.get('save_separate_stepwise_checkpoints', False):
                        to_save = {
                            'n_steps_done': n_steps_done,
                            'model_state_dict': state_dict,
                            'loss': cur_loss,
                        }
                        if config.train.get('save_optimizer_in_all_checkpoints', False):
                            to_save['optimizer_state_dict'] = optimizer.state_dict()
                        torch.save(to_save, checkpoint_path)

                    torch.save({
                        'n_steps_done': n_steps_done,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': cur_loss,
                    }, os.path.join(config.train.save_dir, 'model_latest.pt'))
                    print(f"Saved checkpoint at step {n_steps_done} to {checkpoint_path}")
                else:
                    print("Skipping checkpoint save (no_save_weights=True)")
    
    if is_master:
        print("Training completed!")
        print("Max memory allocated:", torch.cuda.max_memory_allocated(device) / (1024 ** 2), "MB")
        print("Max memory reserved:", torch.cuda.max_memory_reserved(device) / (1024 ** 2), "MB")
        print("Saved in " , config.train.save_dir)
    if use_ddp:
        torch.distributed.destroy_process_group()
    return model



