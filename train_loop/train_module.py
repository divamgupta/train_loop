import sys
sys.path.append(".")

import os
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_descriptor')

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
from .utils.dynamic_import import build_class, get_obj
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils.git import save_git_state
from .optimizer import get_opt
from .utils.model_loading import save_checkpoint, resume_from_checkpoint
from .utils.model_utils import move_to_device
from .utils.device_utils import validate_device, get_ddp_backend, get_autocast_device_type
from .evaluate_module import evaluate_loss
from .utils.download import download_file
from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from types import SimpleNamespace
import datetime

try:
    from accelerate import Accelerator
except ImportError:
    print("Accelerate not installed. To use accelerate features, please install it via 'pip install accelerate'")
    Accelerator = None


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


def crash_detect(crash_detect_config, metrics_dict):
    for k , v in crash_detect_config.items():
        if k in metrics_dict:
            metric_value = metrics_dict[k]
            if 'max' in v:
                if metric_value > v['max']:
                    print(f"Crash detected: Metric {k} = {metric_value} exceeds max {v['max']}")
                    return True
            if 'min' in v:
                if metric_value < v['min']:
                    print(f"Crash detected: Metric {k} = {metric_value} below min {v['min']}")
                    return True
    return False


def train(config):

    validate_device(config.train.device, rocm=config.train.get("rocm", False))

    if config.train.pt_single_threaded:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    if 'resume_epoch_num' in config.train or 'epochs' in config.train or 'no_save_epoch_wise_weights' in config.train or 'num_per_epoch' in config.train:
        raise NotImplementedError("Epoch-based training is not supported in this training loop. Please use step-based training with 'n_total_steps' or 'n_total_samples'.")

    # DDP support
    use_ddp = config.train.use_ddp
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if use_ddp else 0
    world_size = int(os.environ.get('WORLD_SIZE', 1)) if use_ddp else 1
    n_gpus = config.train.n_gpus
    if n_gpus == 'auto':
        n_gpus = torch.cuda.device_count()

    gradient_accum_steps = config.train.gradient_accum_steps

    effective_batch_size = config.train.batch_size * n_gpus * gradient_accum_steps

    find_unused_parameters = config.train.find_unused_parameters

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
        config.train.training_outs_log_frequency = 5
        config.train.log_frequency = 3

        # if exists delete the dir 
        if is_master:
            if os.path.exists(config.train.save_dir):
                shutil.rmtree(config.train.save_dir)

        if use_ddp:
            time.sleep(2)  # Ensure all processes wait for dir deletion

    accelerator = None
    if config.train.use_accelerate:
        accelerator = Accelerator(mixed_precision=config.train.accelerate_mixed_precision)

    is_master = accelerator.is_main_process if accelerator is not None else is_master


    n_total_steps = config.train.n_total_steps
    n_total_samples = config.train.n_total_samples
    n_steps_done = 0

    if n_total_steps < 0 and n_total_samples < 0:
        raise ValueError("Either 'n_total_steps' or 'n_total_samples' must be specified and greater than 0.")

    if n_total_steps < 0:
        n_total_steps = n_total_samples // effective_batch_size

    

    is_compile_model = config.train.compile_model
    is_compile_model_with_loss = config.train.is_compile_model_with_loss
    is_crashed = False

    grad_clip_value = config.train.grad_clip_value

    if is_compile_model_with_loss:
        assert is_compile_model, "If compiling model with loss, 'compile_model' must be True."


    if config.train.save_dir is not None:
        out_config_path = os.path.join(config.train.save_dir, 'config.yaml')
        if os.path.exists(out_config_path) and (not config.train.resume):
            raise FileExistsError(f"Config file {out_config_path} already exists. Please remove it or choose a different save directory.")
    
    use_tqdm = config.train.use_tqdm
    if use_ddp:
        use_tqdm = False

    if accelerator is None:
        if use_ddp:
            _ddp_backend = get_ddp_backend(
                config.train.get("ddp_backend", "auto"),
                config.train.device,
            )
            print(f"[train_loop] DDP backend: {_ddp_backend}")
            torch.distributed.init_process_group(backend=_ddp_backend, timeout=datetime.timedelta(minutes=config.train.nccl_timeout_minutes)  )
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
    else:
        device = accelerator.device
   

    is_distributed = use_ddp or n_gpus > 1

    _cfg_ac = config.train.get("autocast_device_type", "auto")
    _ac_device_type = (
        _cfg_ac if _cfg_ac != "auto"
        else get_autocast_device_type(config.train.device)
    )
    if config.train.use_bfloat16_autocast:
        autocast_ctx = torch.amp.autocast(device_type=_ac_device_type, dtype=torch.bfloat16)
    elif config.train.use_float16_autocast:
        autocast_ctx = torch.amp.autocast(device_type=_ac_device_type, dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    if accelerator is not None:
        autocast_ctx = nullcontext()

    if config.download_assets is not None :
        if is_master:
            for _ , asset in config.download_assets.items():
                # if starts with http then asset is a url
                if 'url' in asset:
                    asset_url = asset.url
                    download_file(asset_url)
                    print(f"Downloaded asset from {asset_url}")
                elif 'function' in asset:
                    download_func = get_obj(asset.function)
                    if 'args' in asset:
                        download_func(**asset.args)
                    else:
                        download_func()
                    print(f"Downloaded asset using function {asset.function}")


        if is_distributed:
            torch.distributed.barrier()


    if config.train.create_model_meta_init:
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

    if 'metrics_only_val' in config:
        if 'name' in config.metrics_only_val:
            metrics_only_val_module = build_class(config.metrics_only_val)
        else:
            metrics_only_val_module = LossModule(**config.metrics_only_val)
    else:
        metrics_only_val_module = None

    model_with_loss = None
    if is_compile_model:
        if is_compile_model_with_loss:
            model_with_loss = get_compiled_model_with_loss(model, loss_function, device)
        else:
            model = torch.compile(model, dynamic=False) 
            assert accelerator is None, "Accelerate and torch.compile cannot be used together yet."
    
    if accelerator is None:
        if use_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=find_unused_parameters)
            if model_with_loss is not None:
                model_with_loss = DDP(model_with_loss, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
        elif n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
            if model_with_loss is not None:
                model_with_loss = torch.nn.DataParallel(model_with_loss, device_ids=list(range(n_gpus)))

    model_module = model.module if (is_distributed and accelerator is None) else model

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
            if accelerator is None:
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
        if accelerator is None:
            if use_ddp:
                gan_loss = DDP(gan_loss, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
            elif n_gpus > 1:
                gan_loss = torch.nn.DataParallel(gan_loss, device_ids=list(range(n_gpus)))
    gan_loss_module = None
    
    if gan_loss is not None:
        gan_loss_module = gan_loss.module if (is_distributed and accelerator is None) else gan_loss
    
    dataset = None
    if is_distributed:
        if is_master:
            print("Loading dataset on master...")
            dataset = build_class(config.dataset) # first just get dataset on master ( so that it can download if needed)
            print("Dataset loaded on master.")
        torch.distributed.barrier()
        if not is_master:
            print(f"Loading dataset on rank {local_rank}...")
            dataset = build_class(config.dataset) # then get dataset on other ranks
            print(f"Dataset loaded on rank {local_rank}.")
        torch.distributed.barrier()
    else:
        dataset = build_class(config.dataset)
    
    # Setup dataloader
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    if use_ddp and accelerator is None:
        train_sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                               collate_fn=collate_fn, num_workers=config.train.num_workers, shuffle=False, sampler=train_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                               collate_fn=collate_fn, num_workers=config.train.num_workers, shuffle=True)

    # Initialize validation dataset if specified in config
    val_dataset = None
    val_dataloader = None
    if is_master:
        if 'val_dataset' in config:
            val_dataset = build_class(config.val_dataset)
            val_collate_fn = val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None
            # since validation happens on master, we do not use DistributedSampler
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=config.train.get('val_batch_size', config.train.batch_size),
                collate_fn=val_collate_fn, 
                num_workers=config.train.get('val_num_workers', config.train.num_workers),
                shuffle=False
            )

    dataset.extra_models = extra_models
    if val_dataset is not None:
        val_dataset.extra_models = extra_models

    # Load checkpoint if specified
    if config.train.get_optimizer_fn_name is not None:
        get_opt_fn_name = config.train.get_optimizer_fn_name
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
        disc_optimizer = get_opt(gan_loss_module, config.train.gan_optimizer, print_summary=is_master)
    
    # Handle checkpoint resuming
    n_steps_done, _ = resume_from_checkpoint( config.train, is_master=is_master , device=device , optimizer=optimizer, model_module=model_module, gan_loss_module=gan_loss_module, disc_optimizer=disc_optimizer)
    

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
    if is_master and config.train.save_dir is not None: # not use_ddp or local_rank == 0:
        os.makedirs(config.train.save_dir, exist_ok=True)
        # Save git state for reproducibility
        save_git_state(config.train.save_dir)
        # Save the final config to the run directory
        out_config_path = os.path.join(config.train.save_dir, 'config.yaml')
        with open(out_config_path, 'w') as f:
            OmegaConf.save(config, f)

    # ── Weights & Biases ────────────────────────────────────────────────────
    wandb_run = None
    if is_master and 'wandb' in config and config.wandb is not None and config.wandb.get('project') is not None:
        import wandb as wandb_lib
        wandb_cfg = OmegaConf.to_container(config.wandb, resolve=True)
        api_key = wandb_cfg.pop('api_key', None) or os.environ.get('WANDB_API_KEY')
        if api_key:
            wandb_lib.login(key=api_key)
        wandb_name = wandb_cfg.get('name')
        if config.get('sanity', False):
            wandb_name = f"__sanity__{wandb_name or ''}"
        # Resume existing W&B run if ID was saved from a previous run
        wandb_id = None
        if config.train.save_dir is not None:
            wandb_id_file = os.path.join(config.train.save_dir, 'wandb_id.txt')
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, 'r') as f:
                    wandb_id = f.read().strip()
                print(f"Resuming W&B run: {wandb_id}")
        wandb_run = wandb_lib.init(
            id=wandb_id,
            resume="allow" if wandb_id else None,
            project=wandb_cfg.get('project'),
            name=wandb_name,
            entity=wandb_cfg.get('entity'),
            tags=wandb_cfg.get('tags'),
            config=OmegaConf.to_container(config, resolve=True),
        )
        # Save W&B run ID for future resumption
        if config.train.save_dir is not None:
            wandb_id_file = os.path.join(config.train.save_dir, 'wandb_id.txt')
            with open(wandb_id_file, 'w') as f:
                f.write(wandb_run.id)
        print(f"W&B run initialised: {wandb_run.url}")

    # Put models in training/eval mode
    model.train()

    # If extra models and gan_loss, set train mode
    if gan_loss is not None:
        gan_loss.train()
    

    if is_distributed:
        torch.distributed.barrier()

    summary_frequency = config.train.summary_frequency
    eval_frequency = config.train.num_eval_every_steps
    log_frequency = config.train.log_frequency
    rolling_window = config.train.metrics_rolling_window
    
    scaler = None
    if config.train.use_grad_scaler:
        # bfloat16 has sufficient dynamic range and does not need loss scaling.
        _use_scaler = config.train.use_float16_autocast and not config.train.use_bfloat16_autocast
        scaler = GradScaler(enabled=_use_scaler)

    if use_tqdm:
        progress_bar = tqdm(range(0, n_total_steps), desc=f"Step {n_steps_done}/{n_total_steps} - Avg Loss: 0.000000")
    else:
        progress_bar = None

    iter_times = []
    loss_history = {}


    training_outs_log_functions = []

    if is_master:
        if "training_outs_log_functions" in config and config.training_outs_log_functions is not None:
            for func in config.training_outs_log_functions:
                log_func = build_class(func)
                training_outs_log_functions.append(log_func)

    if accelerator is not None:
        model, optimizer, dataloader, lr_scheduler, gan_loss, val_dataloader,disc_optimizer = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler, gan_loss, val_dataloader, disc_optimizer
        )
        for k in extra_models:
            extra_models[k] = accelerator.prepare(extra_models[k])

    batch_inputs_dict = None

    # Training loop
    while True: # dw it breaks when n_steps_done >= n_total_steps

        if is_distributed:
            is_crashed = bool((lambda t: torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX) or t.item())(torch.tensor(int(is_crashed), device=device)))
        
        if is_crashed:
            if config.train.crash_recovery_mode == "exit":
                assert False, "Training crashed ."
            elif config.train.crash_recovery_mode == "resume_from_latest_checkpoint":
                n_steps_done , was_loaded = resume_from_checkpoint( SimpleNamespace(
                    resume=True,
                    save_dir=config.train.save_dir,
                    resume_checkpoint_path=None,
                    resume_steps_num=None
                ), is_master=is_master , device=device , optimizer=optimizer, model_module=model_module, gan_loss_module=gan_loss_module, disc_optimizer=disc_optimizer)
                if not was_loaded:
                    raise RuntimeError("Training crashed no checkpoint found to resume from.")
                is_crashed = False
            else:
                raise ValueError(f"Unknown crash_recovery_mode: {config.train.crash_recovery_mode}")

        if summary_frequency > 0 and  n_steps_done % summary_frequency == 0:
            if is_distributed:
                torch.distributed.barrier()
            if is_master:
                if "summary_functions" in config and config.summary_functions is not None:
                    for func in config.summary_functions:
                        summary_func = build_class(func)
                        with autocast_ctx:
                            # Pass save_dir (could be None) but let summary_func decide what to do
                            summary_func.run(model_module, (val_dataset if val_dataset is not None else dataset), config.train.save_dir, n_steps_done)
        

        iter_start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if gradient_accum_steps < 1:
            raise ValueError("gradient_accum_steps must be >= 1")

        for mini_i in range(gradient_accum_steps):

            if config.train.debug_ddp_sync:
                print("mini_i:", mini_i, "rank:", local_rank)
                if is_master:
                    time.sleep(5)
            
            if config.train.debug_load_batch_only_once:
                if batch_inputs_dict is None:
                    batch_inputs_dict = next(data_generator)
            else:
                batch_inputs_dict = next(data_generator)

            if batch_inputs_dict is None:
                print("Batch is None, skipping...")

                if use_ddp:
                    raise ValueError("Dataloader returned None will make it go out of sync.")

                continue
            
            if accelerator is None: # accelerator automatically handles moving to device
                batch_inputs_dict = move_to_device(batch_inputs_dict, device)

            if hasattr(dataset , 'post_process_batch'):
                dataset.post_process_batch(batch_inputs_dict)

            only_sync_end_ctx = nullcontext()
            if is_distributed:
                if (mini_i < gradient_accum_steps -1):
                    if accelerator is not None:
                        only_sync_end_ctx = accelerator.no_sync(model) 
                    else:
                        only_sync_end_ctx = model.no_sync() 
            
            with autocast_ctx, only_sync_end_ctx:

                losses = {}
                model_outputs = None
                model_outputs_detached = None
                if model_with_loss is not None:
                    loss, losses = model_with_loss(batch_inputs_dict)
                else:
                    model_outputs = model(batch_inputs_dict)
                    model_outputs_detached = { k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in model_outputs.items() }

                if is_master and mini_i == 0 and len(training_outs_log_functions) > 0 and (n_steps_done % config.train.training_outs_log_frequency == 0):
                    for log_func in training_outs_log_functions:
                        log_func.run(model_module, dataset, config.train.save_dir, n_steps_done, model_outputs , batch_inputs_dict)

                if gan_loss is not None:
                    assert gradient_accum_steps == 1, "GAN loss with gradient accumulation is not supported yet."
                    # For DataParallel, access .module if needed                    
                    assert model_outputs_detached is not None, "Model outputs should not be None when using GAN loss."
                    loss_disc = gan_loss_module.discriminator_loss(batch_inputs_dict, model_outputs_detached)
                    if torch.any(torch.isnan(loss_disc)):
                        print("NaN detected in loss at step ", n_steps_done)
                        is_crashed = True
                    disc_optimizer.zero_grad()
                    if accelerator is not None:
                        accelerator.backward(loss_disc)
                        disc_optimizer.step()
                    else:
                        if scaler is None:
                            loss_disc.backward(retain_graph=True)
                            disc_optimizer.step()
                        else:
                            scaler.scale(loss_disc).backward(retain_graph=True)
                            scaler.step(disc_optimizer)
                            scaler.update()

                    
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
                    gen_loss = gan_loss_module.generator_loss(batch_inputs_dict, model_outputs)
                    loss += gen_loss * config.gan_loss.gen_loss_weight
                    losses['gen_loss'] = gen_loss
                    losses['disc_loss'] = loss_disc

            if accelerator is not None:
                accelerator.backward(loss / gradient_accum_steps)
            else:
                if scaler is  None:
                    (loss / gradient_accum_steps).backward()
                else:
                    scaler.scale(loss / gradient_accum_steps).backward()

        if torch.any(torch.isnan(loss)):
            print("NaN detected in loss at step ", n_steps_done)
            is_crashed = True

        grad_metrics = {}
        if "grad_metrics" in config and config.grad_metrics is not None:
            for grad_metric_name , grad_metric_conf in config.grad_metrics.items():
                grad_metric_conf_copy = grad_metric_conf.copy()
                grad_metric_fn_name = grad_metric_conf_copy.pop("function_name", grad_metric_name)
                if "." not in grad_metric_fn_name:
                    grad_metric_fn_name = "train_loop.losses." + grad_metric_fn_name
                grad_metric_fn = get_obj(grad_metric_fn_name )
                grad_metrics[grad_metric_name] = grad_metric_fn(model=model_module,**grad_metric_conf_copy)

        grad_clip_enabled = grad_clip_value > 0.0
        if grad_clip_enabled:
            if accelerator is None:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                grad_norm = grad_norm_tensor  # GPU tensor -> CPU float (note: cpu-gpu sync point)
            else:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip_value)

        if lr_scheduler is not None:
            lr_scheduler.step(n_steps_done)

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        optimizer.zero_grad()

        optimizer_metrics = {}
        if "optimizer_metrics" in config and config.optimizer_metrics is not None:
            for opt_metric_name , opt_metric_conf in config.optimizer_metrics.items():
                opt_metric_conf_copy = opt_metric_conf.copy()
                opt_metric_fn_name = opt_metric_conf_copy.pop("function_name" , opt_metric_name)
                if "." not in opt_metric_fn_name:
                    opt_metric_fn_name = "train_loop.losses." + opt_metric_fn_name
                opt_metric_fn = get_obj(opt_metric_fn_name )
                optimizer_metrics[opt_metric_name] = opt_metric_fn(optimizer=optimizer,**opt_metric_conf_copy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        n_steps_done += 1
        
        iter_end_time = time.time()
        
        # Update rolling average for each loss component
        losses_plus_metrics = losses.copy()
        losses_plus_metrics.update(metrics_result or {})
        losses_plus_metrics.update(grad_metrics)
        losses_plus_metrics.update(optimizer_metrics)
        losses_plus_metrics['total_loss'] = loss 

        if config.train.log_iter_time:
            losses_plus_metrics['iter_time'] = iter_end_time - iter_start_time

        if grad_clip_enabled:
            losses_plus_metrics['grad_norm'] = grad_norm

        if config.train.crash_detect_params is not None:
            if crash_detect( config.train.crash_detect_params , losses_plus_metrics):
                is_crashed = True

        for k, v in losses_plus_metrics.items():
            if k not in loss_history:
                loss_history[k] = []
            loss_history[k].append( v.item() if isinstance(v, torch.Tensor) else v )
            if len(loss_history[k]) > rolling_window:
                loss_history[k].pop(0)
        
        # Log training loss to jsonl file every rolling_window steps
        if n_steps_done % log_frequency == 0:
            train_loss_components = {k: sum(v)/len(v) for k, v in loss_history.items()}
            train_loss_components['step'] = n_steps_done
            if config.train.save_dir is not None:
                train_loss_file = os.path.join(config.train.save_dir, 'train_loss.jsonl')
                with open(train_loss_file, 'a') as f:
                    f.write(json.dumps(train_loss_components) + '\n')
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in train_loss_components.items() if k != 'step'}, step=n_steps_done)
        
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
                        val_loss, val_loss_components = evaluate_loss(model_module, val_dataloader, val_loss_fn, device, num_val_steps=config.train.num_validation_steps, use_tqdm=config.train.validation_use_tqdm)
                        # get metrics also 
                        if metrics_module is not None:
                            metrics_fn = metrics_module
                            if is_distributed and hasattr(metrics_module, 'module'):
                                metrics_fn = metrics_module.module
                            _ , val_metrics = evaluate_loss(model_module, val_dataloader, metrics_fn, device, num_val_steps=config.train.num_validation_steps, use_tqdm=config.train.validation_use_tqdm)
                            val_loss_components.update(val_metrics)
                        if metrics_only_val_module is not None:
                            metrics_only_val_fn = metrics_only_val_module
                            if is_distributed and hasattr(metrics_only_val_module, 'module'):
                                metrics_only_val_fn = metrics_only_val_module.module
                            _ , val_metrics_only = evaluate_loss(model_module, val_dataloader, metrics_only_val_fn, device, num_val_steps=config.train.num_validation_steps, use_tqdm=config.train.validation_use_tqdm)
                            val_loss_components.update(val_metrics_only)
                            
                        val_loss_components['total_loss'] = val_loss
                        val_loss_components['step'] = n_steps_done
                        if config.train.save_dir is not None:
                            val_loss_file = os.path.join(config.train.save_dir, 'val_loss.jsonl')
                            with open(val_loss_file, 'a') as f:
                                f.write(json.dumps(val_loss_components) + '\n')
                        print(f"Validation Loss at step {n_steps_done}: {val_loss:.6f}")
                        print(f"Validation Loss Components: {val_loss_components}")
                        if wandb_run is not None:
                            wandb_run.log({f"val/{k}": v for k, v in val_loss_components.items() if k != 'step'}, step=n_steps_done)

            if is_distributed:
                torch.distributed.barrier()
        
        # Calculate rolling average for each loss component
        loss_str = " | ".join([f"{k}: {sum(history)/len(history):.4f}" for k, history in loss_history.items()])
        if use_tqdm:
            progress_bar.set_description(f"Step {n_steps_done}/{n_total_steps} -  {loss_str}")
            progress_bar.n = n_steps_done
            progress_bar.refresh()
        else:
            iter_time = iter_end_time - iter_start_time
            iter_times.append(iter_time)
            avg_iter_time = sum(iter_times) / len(iter_times)
            remaining_iters = n_total_steps - n_steps_done
            eta = remaining_iters * avg_iter_time
            print(f"Rank {local_rank} Step {n_steps_done}/{n_total_steps} - {loss_str} | Iter time: {iter_time:.2f}s | ETA: {eta:.2f}s")
        

        # Save checkpoint
        exact_save_iters = config.train.checkpoint_save_iterations
        is_exact_save = exact_save_iters is not None and n_steps_done in exact_save_iters
        is_freq_save = n_steps_done % config.train.checkpoint_save_frequency == 0
        if (is_freq_save or is_exact_save or n_steps_done >= n_total_steps) and not is_crashed:
            save_checkpoint(config.train, is_master, optimizer=optimizer, model_module=model_module, n_steps_done=n_steps_done, loss_history=loss_history, gan_loss_module=gan_loss_module, disc_optimizer=disc_optimizer, force_stepwise_save=is_exact_save)

        if n_steps_done >= n_total_steps:
            break
    
    if is_master:
        print("Training completed!")
        if 'cuda' in str(device) and torch.cuda.is_available():
            print("Max memory allocated:", torch.cuda.max_memory_allocated(device) / (1024 ** 2), "MB")
            print("Max memory reserved:", torch.cuda.max_memory_reserved(device) / (1024 ** 2), "MB")
        if config.train.save_dir is not None:
            print("Saved in " , config.train.save_dir)
        if wandb_run is not None:
            wandb_run.finish()
    if use_ddp:
        torch.distributed.destroy_process_group()
    return model



