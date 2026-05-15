# train_loop — Comprehensive Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Configuration Reference](#5-configuration-reference)
6. [Model Interface](#6-model-interface)
7. [Dataset Interface](#7-dataset-interface)
8. [Loss Functions & Metrics](#8-loss-functions--metrics)
9. [Optimizer Configuration](#9-optimizer-configuration)
10. [Learning Rate Schedulers](#10-learning-rate-schedulers)
11. [Distributed Training](#11-distributed-training)
12. [Mixed Precision](#12-mixed-precision)
13. [Model Compilation](#13-model-compilation)
14. [Gradient Accumulation](#14-gradient-accumulation)
15. [Checkpointing & Resuming](#15-checkpointing--resuming)
16. [GAN Training](#16-gan-training)
17. [Crash Detection & Recovery](#17-crash-detection--recovery)
18. [OmegaConf Features](#18-omegaconf-features)
19. [GPU Selection](#19-gpu-selection)
20. [Asset Downloads](#20-asset-downloads)
21. [Sanity Check Mode](#21-sanity-check-mode)
22. [Summary & Logging Functions](#22-summary--logging-functions)
23. [Tutorials](#23-tutorials)
24. [Python API](#24-python-api)
25. [CLI Reference](#25-cli-reference)
26. [Weights & Biases Integration](#26-weights--biases-integration)
27. [Dataset Demo](#27-dataset-demo)


---

## 1. Overview

**train_loop** is a configuration-driven PyTorch training framework that eliminates boilerplate training code. You provide:

- A YAML config file specifying your model, dataset, losses, and training hyperparameters
- A `nn.Module` subclass that accepts a dict as input and returns a dict as output
- A `torch.utils.data.Dataset` subclass that returns dicts

The framework handles the rest: the training loop, checkpointing, validation, distributed training, mixed precision, model compilation, GAN training, crash detection, and more.

**Why train_loop?**

- Write your model and dataset once — no training loop boilerplate
- Scale from single GPU to multi-node DDP by changing two config lines
- Mix and match features (grad scaling, compilation, cosine LR, param groups) declaratively
- Reproduce experiments by diffing YAML files

---

## 2. Installation

train_loop has no package installer at this time. Run it directly from the repository root:

```bash
git clone <repo-url>
cd train_loop
pip install torch torchvision omegaconf tqdm
# Optional extras:
pip install accelerate   # for accelerate integration
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- omegaconf
- tqdm

---

## 3. Quick Start

**Step 1 — Define your model** (`my_model.py`):

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Linear(784, hidden)
        self.head = nn.Linear(hidden, 10)

    def forward(self, batch):          # batch is a dict
        x = batch["inp"]               # pull tensors from dict
        x = torch.relu(self.net(x))
        return {"logits": self.head(x)} # return a dict
```

**Step 2 — Define your dataset** (`my_dataset.py`):

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, split="train"):
        # load data...
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return {"inp": torch.randn(784), "label": torch.randint(0, 10, ())}
```

**Step 3 — Write a config** (`config.yml`):

```yaml
model:
  name: my_model.MyModel
  args:
    hidden: 256

dataset:
  name: my_dataset.MyDataset
  args:
    split: train

val_dataset:
  name: my_dataset.MyDataset
  args:
    split: val

losses:
  ce:
    function_name: cross_entropy_classification_loss
    weight: 1.0
    src_key: logits
    tgt_key: label

train:
  batch_size: 64
  n_total_steps: 5000
  save_dir: /tmp/my_run
  device: cuda
  optimizer:
    name: adam
    args:
      lr: 1e-3
```

**Step 4 — Train:**

```bash
python train.py config.yml
```

That's it. Checkpoints, logs, and the resolved config are written to `save_dir`.

---

## 4. Core Concepts

### The Dict Contract

Everything in train_loop communicates through Python dicts:

```
DataLoader  →  batch (dict)  →  Model  →  model_outs (dict)
                                              ↓
                                        Loss functions
                                          (batch + model_outs → scalar)
```

- **Dataset** `__getitem__` returns a dict, e.g. `{"inp": tensor, "label": tensor}`
- **Model** `forward(batch)` receives the batch dict and returns an output dict, e.g. `{"logits": tensor}`
- **Loss functions** receive `(batch, model_outs)` and return a scalar tensor

This uniform interface means every built-in feature (mixed precision, DDP, compilation, GAN) works with any model and dataset combination.

### Step-Based Training

train_loop is **step-based, not epoch-based**. Specify either:
- `train.n_total_steps`: total number of optimizer steps
- `train.n_total_samples`: total samples; steps = `n_total_samples // effective_batch_size`

The dataloader loops infinitely; training ends when the step count is reached.

### Dotpath Class Loading

All class references in the config use Python dotpath notation:

```yaml
model:
  name: examples.mnist.mnist.MNISTClassifier  # module.path.ClassName
```

This resolves via `importlib` relative to the working directory. Always run `python train.py` from the repository root.

---

## 5. Configuration Reference

The config is loaded with [OmegaConf](https://omegaconf.readthedocs.io/) and merged with the default config. Every key below is optional unless noted; defaults come from `train_loop/default_config.py`.

### Top-Level Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | dict | — | Model class and constructor args (required) |
| `model.name` | str | `null` | Dotpath to `nn.Module` subclass |
| `model.args` | dict | `{}` | Keyword arguments passed to model constructor |
| `model.ckpt_path` | str\|null | `null` | Load pre-trained weights before training |
| `dataset` | dict | — | Training dataset class and args (required) |
| `dataset.name` | str | `null` | Dotpath to `Dataset` subclass |
| `dataset.args` | dict | `{}` | Keyword arguments for dataset constructor |
| `val_dataset` | dict | — | Validation dataset (optional; same schema as `dataset`) |
| `losses` | dict | — | Loss function definitions (required) |
| `metrics` | dict | — | Metric definitions; computed during train and validation |
| `metrics_only_val` | dict | — | Metrics computed only during validation |
| `optimizer_metrics` | dict | — | Functions taking `optimizer=` as argument (e.g., learning rate) |
| `grad_metrics` | dict | — | Functions taking `model=` as argument (e.g., gradient norm) |
| `summary_functions` | list | `null` | Classes with `.run(model, dataset, out_dir, iter)` called periodically |
| `training_outs_log_functions` | list | `null` | Classes with `.run(model, dataset, out_dir, iter, model_outputs, batch)` |
| `gan_loss` | dict\|null | `null` | GAN loss configuration |
| `extra_models` | dict | `null` | Additional frozen models passed to dataset via `dataset.extra_models` |
| `download_assets` | dict | `{}` | Files to download before training starts |
| `gpus` | list\|str\|null | `null` | GPU IDs to use; `"auto"` for automatic selection |
| `sanity` | bool | `false` | Run a short sanity-check training with fixed settings |
| `train` | dict | — | All training hyperparameters (see below) |

### `train.*` Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device` | str | `"cuda"` | Device string (`"cuda"`, `"cpu"`, `"cuda:0"`) |
| `batch_size` | int | `2` | Per-GPU batch size |
| `n_total_steps` | int | `-1` | Total optimizer steps; use `-1` when using `n_total_samples` |
| `n_total_samples` | int | `-1` | Total samples; derives `n_total_steps` automatically |
| `save_dir` | str\|null | `null` | Directory for checkpoints, logs, and config snapshot |
| `optimizer` | dict | `{name: adam}` | Optimizer configuration (see [Section 9](#9-optimizer-configuration)) |
| `lr_scheduler` | dict | — | LR scheduler configuration (see [Section 10](#10-learning-rate-schedulers)) |
| `gradient_accum_steps` | int | `1` | Gradient accumulation steps |
| `grad_clip_value` | float | `0` | Max gradient norm; `0` disables clipping |
| `use_ddp` | bool | `false` | Enable DistributedDataParallel |
| `n_gpus` | int\|`"auto"` | `1` | Number of GPUs; `"auto"` uses all available |
| `find_unused_parameters` | bool | `true` | DDP `find_unused_parameters` flag |
| `compile_model` | bool | `false` | Compile model with `torch.compile` |
| `is_compile_model_with_loss` | bool | `false` | Compile model+loss together (requires `compile_model: true`) |
| `use_bfloat16_autocast` | bool | `false` | Enable bfloat16 autocast |
| `use_float16_autocast` | bool | `false` | Enable float16 autocast |
| `use_grad_scaler` | bool | `false` | Enable GradScaler (for float16) |
| `use_accelerate` | bool | `false` | Use HuggingFace Accelerate |
| `accelerate_mixed_precision` | str\|null | `null` | `"bf16"`, `"fp16"`, or `null` |
| `resume` | bool | `false` | Auto-resume from latest checkpoint in `save_dir` |
| `resume_checkpoint_path` | str\|null | `null` | Path or URL to specific checkpoint |
| `resume_steps_num` | int\|null | `null` | Override step count when loading a checkpoint |
| `checkpoint_save_frequency` | int | `1000` | Save checkpoint every N steps |
| `checkpoint_save_iterations` | list\|null | `null` | Exact iterations to force-save a `model_step_N.pt` (independent of frequency) |
| `save_separate_stepwise_checkpoints` | bool | `false` | Save `model_step_N.pt` in addition to `model_latest.pt` |
| `save_optimizer_in_all_checkpoints` | bool | `false` | Include optimizer state in every stepwise checkpoint |
| `no_save_weights` | bool | `false` | Disable all checkpoint saving |
| `num_eval_every_steps` | int | `1000` | Run validation every N steps |
| `num_validation_steps` | int | `-1` | Max validation batches; `-1` = full val set |
| `validation_use_tqdm` | bool | `false` | Show tqdm progress bar during validation |
| `summary_frequency` | int | `1000` | Call `summary_functions` every N steps |
| `training_outs_log_frequency` | int | `1000` | Call `training_outs_log_functions` every N steps |
| `log_frequency` | int | `1` | Write loss to `train_loss.jsonl` every N steps |
| `metrics_rolling_window` | int | `1` | Rolling window size for loss display |
| `num_workers` | int | `0` | DataLoader worker processes |
| `use_tqdm` | bool | `false` | Show tqdm progress bar (disabled automatically under DDP) |
| `pt_single_threaded` | bool | `true` | Set PyTorch to single-threaded mode |
| `log_iter_time` | bool | `false` | Include iteration time in logged metrics |
| `nccl_timeout_minutes` | int | `60` | NCCL communication timeout for DDP |
| `crash_detect_params` | dict\|null | `null` | Metric thresholds that trigger crash detection |
| `crash_recovery_mode` | str | `"resume_from_latest_checkpoint"` | Action on crash: `"exit"`, `"resume_from_latest_checkpoint"`, `"reinit"` |
| `create_model_meta_init` | bool | `false` | Initialize model on `meta` device (memory-efficient for large models) |
| `debug_load_batch_only_once` | bool | `false` | Reuse the first batch every step (debugging) |
| `debug_ddp_sync` | bool | `false` | Print DDP sync points for debugging |
| `get_optimizer_fn_name` | str\|null | `null` | Call `model.<fn>()` to get the optimizer instead of config |
| `gan_optimizer` | dict | — | Discriminator optimizer (required when `gan_loss` is set) |

---

## 6. Model Interface

A train_loop model is a standard `nn.Module` with one constraint: **`forward` receives a batch dict and must return a dict**.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # ... define layers ...

    def forward(self, batch: dict) -> dict:
        # Pull inputs from the batch dict
        x = batch["inp"]

        # ... compute ...

        # Return predictions as a dict
        return {"pred": output}
```

### Config Specification

```yaml
model:
  name: my_module.MyModel   # dotpath to class
  args:                     # passed as **kwargs to __init__
    hidden: 256
  ckpt_path: /path/to/weights.pt  # optional pre-trained weights
```

### Optional Hook: `init_weights`

If the model has an `init_weights()` method, it is called after the model is moved to the device:

```python
def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
```

### Memory-Efficient Init: `create_model_meta_init`

For very large models, setting `train.create_model_meta_init: true` initializes the model on the `meta` device first (no memory allocated), then moves it to GPU with `to_empty(device=device)`. Requires the model's `init_weights` to fill the parameters:

```yaml
train:
  create_model_meta_init: true
```

### Extra Models

Frozen auxiliary models can be attached to your dataset at runtime:

```yaml
extra_models:
  encoder:
    name: my_module.Encoder
    args:
      pretrained: true
```

These are available as `dataset.extra_models["encoder"]` inside `__getitem__` or `post_process_batch`.

---

## 7. Dataset Interface

A train_loop dataset is a standard `torch.utils.data.Dataset` that **returns dicts**:

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, split="train"):
        # ... load data ...
        self.collate_fn = None  # optional custom collate function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        sample = self.data[idx]
        return {
            "inp": sample["image"],    # input tensor
            "label": sample["class"],  # target tensor
        }
```

### Config Specification

```yaml
dataset:
  name: my_module.MyDataset
  args:
    split: train

val_dataset:
  name: my_module.MyDataset
  args:
    split: val
```

### Custom Collate Function

Set `self.collate_fn` on the dataset instance to override the default PyTorch collation:

```python
class MyDataset(Dataset):
    def __init__(self):
        self.collate_fn = my_custom_collate
```

### Post-Processing Batches

Implement `post_process_batch(batch_dict)` for any processing that needs to happen after collation and before the model forward pass:

```python
class MyDataset(Dataset):
    def post_process_batch(self, batch):
        batch["inp"] = batch["inp"].float() / 255.0
```

### Validation Dataset

`val_dataset` uses the same interface. Validation runs on the master process only, without a DistributedSampler.

---

## 8. Loss Functions & Metrics

### Built-in Loss Functions

All built-in functions have the signature `fn(batch, model_outs, src_key, tgt_key, ...)` and are referenced by name in the config. For `tgt_key`, the framework first looks in `batch`; if not found, it looks in `model_outs`.

#### `bce`
Binary cross-entropy. Expects sigmoid-activated predictions.

```yaml
losses:
  my_loss:
    function_name: bce
    src_key: pred_sigmoid   # key in model_outs (sigmoid already applied)
    tgt_key: target         # key in batch or model_outs
    weight: 1.0
    # Optional:
    mask_key: mask          # key for binary mask tensor
```

#### `mse`
Mean squared error.

```yaml
losses:
  reconstruction:
    function_name: mse
    src_key: reconstructed
    tgt_key: original
    weight: 1.0
    # Optional: mask_key
```

#### `l1`
L1 / mean absolute error.

```yaml
losses:
  l1_loss:
    function_name: l1
    src_key: predicted
    tgt_key: ground_truth
    # Optional: mask_key
```

#### `smooth_l1`
Smooth L1 (Huber) loss.

```yaml
losses:
  huber:
    function_name: smooth_l1
    src_key: predicted
    tgt_key: ground_truth
    # Optional: mask_key
```

#### `cross_entropy_classification_loss`
Cross-entropy for classification. Expects raw logits (no softmax).

```yaml
losses:
  cce:
    function_name: cross_entropy_classification_loss
    src_key: pred_logits    # logits in model_outs
    tgt_key: gt_class_id    # integer class IDs in batch
    weight: 1.0
```

#### `accuracy_classification`
Top-k accuracy metric. Use under `metrics:` rather than `losses:`.

```yaml
metrics:
  accuracy:
    function_name: accuracy_classification
    src_key: pred_logits
    tgt_key: gt_class_id
    weight: 1.0
    # Optional:
    top_k: 5   # top-5 accuracy
```

#### `dummy`
Returns `0.0`. Useful when using GAN loss exclusively.

```yaml
losses:
  dummy:
    weight: 1
```

### Loss Config Schema

Each named entry under `losses:` (or `metrics:`) supports:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `function_name` | str | key name | Built-in name or dotpath to a function or class |
| `weight` | float | `1.0` | Multiplier for this loss term |
| `src_key` | str | — | Key to read from `model_outs` |
| `tgt_key` | str | — | Key to read from `batch` (or `model_outs` as fallback) |
| `mask_key` | str | — | Optional mask key (supported by `bce`, `mse`, `l1`, `smooth_l1`) |

### Custom Loss Functions

**Function-based** (simplest):

```python
# my_losses.py
def my_loss(batch, model_outs, src_key, tgt_key):
    pred = model_outs[src_key]
    gt = batch[tgt_key]
    return ((pred - gt) ** 2).mean()
```

```yaml
losses:
  my_custom:
    function_name: my_losses.my_loss
    src_key: pred
    tgt_key: gt
    weight: 0.5
```

**Class-based** (for losses with learnable parameters or state):

```python
# my_losses.py
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=8):
        super().__init__()
        # ... load VGG, etc. ...

    def __call__(self, batch, model_outs):
        # Signature: (batch, model_outs) → scalar tensor
        pred = model_outs["generated"]
        gt = batch["target"]
        return self.compute(pred, gt)
```

```yaml
losses:
  perceptual:
    function_name: my_losses.PerceptualLoss
    weight: 1.0
    feature_layer: 12   # extra kwargs passed to __init__
```

### `metrics` vs `losses`

- `losses:` — used during training for backpropagation; also logged
- `metrics:` — computed during training and validation but **not** added to the backward loss
- `metrics_only_val:` — computed only during validation

All three share the same function/class interface.

### `optimizer_metrics` and `grad_metrics`

These are logged alongside losses but have different function signatures:

```yaml
optimizer_metrics:
  lr:
    function_name: opt_learning_rate  # fn(optimizer) → scalar
    # equivalent dotpath: train_loop.losses.opt_learning_rate

grad_metrics:
  grad_norm:
    function_name: grad_norm          # fn(model, norm_type=...) → scalar
    norm_type: 2.0
```

Built-in functions: `opt_learning_rate(optimizer)`, `grad_norm(model, norm_type=2.0)`.

---

## 9. Optimizer Configuration

### Built-in Optimizers

Specify under `train.optimizer`:

```yaml
train:
  optimizer:
    name: adam          # adam | adamw | sgd | rmsprop
    args:
      lr: 1e-3
      betas: [0.9, 0.999]
      weight_decay: 0.0
```

Supported names: `adam`, `adamw`, `sgd`, `rmsprop`. Any other name is treated as a dotpath to a custom optimizer class.

### Custom Optimizer via Dotpath

```yaml
train:
  optimizer:
    name: my_optimizers.Muon    # dotpath to optimizer class
    args:
      lr: 0.02
      momentum: 0.95
```

### Parameter Groups

To assign different learning rates (or other hyperparameters) to different model submodules:

```yaml
train:
  optimizer:
    name: adamw
    args:
      lr: 1e-4
      weight_decay: 1e-2
    param_groups:
      - params: ["encoder"]    # submodule name or list of submodule/param names
        lr: 1e-5               # overrides global lr for these params
      - params: ["decoder"]
        lr: 1e-3
```

`params` can match:
- An exact parameter name (`"layer.weight"`)
- A submodule name (`"transformer.h"`) — includes all parameters in that submodule recursively

Setting `lr: 0` excludes that group from the optimizer entirely.

### Multi-Optimizer

Use `name: multi` to run separate optimizers over different parameter groups, e.g. to combine Adam for embeddings and a custom optimizer for weight matrices:

```yaml
train:
  optimizer:
    name: multi
    optimizers:
      embeds:
        name: adamw
        args:
          lr: 0.001
          weight_decay: 0.0
        param_groups:
          - params: ["transformer.wte"]
            lr: 0.001
      matrix:
        name: my_opt.Muon
        args:
          lr: 0.02
        param_groups:
          - params: ["transformer.h"]
            lr: 0.02
```

The `MultiOptimizer` wraps all sub-optimizers, calling `zero_grad()` and `step()` on each.

### Optimizer from Model Method

For models that define their own optimizer logic:

```yaml
train:
  get_optimizer_fn_name: __model__.get_optimizer
```

This calls `model.get_optimizer()` and uses the returned optimizer.

---

## 10. Learning Rate Schedulers

### Built-in: `CosineLRScheduler`

Cosine annealing with optional linear warmup. Lives at `train_loop.lr_schedulers.CosineLRScheduler`.

```yaml
train:
  lr_scheduler:
    name: train_loop.lr_schedulers.CosineLRScheduler
    args:
      total_steps: ${train.n_total_steps}  # OmegaConf interpolation
      base_lr: null       # inferred from optimizer if null
      min_lr: 0.0
      warmup_lr: 0.0
      warmup_steps: 0
      epoch_wise: false   # if true, steps_per_epoch is required
```

The scheduler is called each step as `scheduler.step(current_step)` before the optimizer step.

### Custom Scheduler

Implement a class with `step(it)`:

```python
class MyScheduler:
    def __init__(self, optimizer, total_iterations, warmdown_ratio=0.2):
        self.optimizer = optimizer
        # ...

    def step(self, it):
        lr = self.compute_lr(it)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
```

```yaml
train:
  lr_scheduler:
    name: my_module.MyScheduler
    args:
      total_iterations: ${train.n_total_steps}
      warmdown_ratio: 0.2
```

The `optimizer` argument is automatically injected by the framework.

---

## 11. Distributed Training

### DDP (Recommended for Multi-GPU)

Enable DDP with two config lines:

```yaml
train:
  use_ddp: true
  n_gpus: 4
```

train_loop **automatically restarts itself with `torchrun`** if DDP is requested but `LOCAL_RANK` is not set. No manual `torchrun` invocation is needed:

```bash
python train.py config.yml  # → auto-restarts as: torchrun --nproc_per_node=4 ...
```

The NCCL process group is initialized with a configurable timeout:

```yaml
train:
  nccl_timeout_minutes: 60  # default: 60 minutes
```

### DataParallel (Single-Node, No torchrun)

For single-machine multi-GPU without process groups, use `n_gpus > 1` with `use_ddp: false`:

```yaml
train:
  n_gpus: 2
  use_ddp: false   # uses torch.nn.DataParallel
```

This is simpler but typically slower than DDP due to GIL and cross-GPU data transfers.

### `n_gpus: "auto"`

Automatically uses all available CUDA devices:

```yaml
train:
  n_gpus: auto
  use_ddp: true
```

### DDP Dataset Loading

Under DDP, datasets are loaded in order: master process first, then a barrier, then worker processes. This prevents download race conditions when multiple ranks would otherwise try to download data simultaneously.

### Multi-Node Setup

For multi-node training, launch via `torchrun` explicitly and set `LOCAL_RANK` before running, or configure the standard torchrun multi-node flags:

```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=<host> --master_port=29500 \
         train.py config.yml
```

Since `LOCAL_RANK` will be set, the auto-restart logic is bypassed.

### `find_unused_parameters`

If your model has parameters that do not receive gradients on every step:

```yaml
train:
  find_unused_parameters: true  # default: true
```

Disable this for a small performance improvement if all parameters are used on every step.

---

## 12. Mixed Precision

### bfloat16 Autocast (Recommended)

```yaml
train:
  use_bfloat16_autocast: true
```

Wraps the forward pass and loss computation in `torch.amp.autocast(dtype=torch.bfloat16)`. No gradient scaler needed. Works best on Ampere+ GPUs (A100, H100, etc.).

### float16 Autocast + GradScaler

```yaml
train:
  use_float16_autocast: true
  use_grad_scaler: true
```

Uses `torch.amp.autocast(dtype=torch.float16)` with a `GradScaler` to prevent underflow.

### HuggingFace Accelerate

```yaml
train:
  use_accelerate: true
  accelerate_mixed_precision: bf16   # bf16 | fp16 | null
```

Accelerate manages model preparation, optimizer, and dataloader wrapping automatically. Useful for compatibility with HuggingFace ecosystem models.

---

## 13. Model Compilation

### Compile Model Only

```yaml
train:
  compile_model: true
```

Applies `torch.compile(model, dynamic=False)` before training. Typically improves throughput after a short compilation warmup.

### Compile Model + Loss Together

For maximum fusion (eliminates Python overhead between model forward and loss):

```yaml
train:
  compile_model: true
  is_compile_model_with_loss: true
```

This wraps model and loss in a single `nn.Module` and compiles the combined forward. When this is enabled, `model_outputs` are not available to `training_outs_log_functions` during the fused steps.

---

## 14. Gradient Accumulation

Accumulate gradients over multiple mini-batches before updating weights, effectively increasing the batch size without increasing GPU memory:

```yaml
train:
  batch_size: 32
  gradient_accum_steps: 8
  # Effective batch size = 32 * 8 = 256 (single GPU)
  # With n_gpus=4: effective = 32 * 8 * 4 = 1024
```

**Effective batch size formula:**
```
effective_batch_size = batch_size × n_gpus × gradient_accum_steps
```

Use OmegaConf expressions to compute `gradient_accum_steps` from a target effective batch size:

```yaml
effective_batch_size: 256

train:
  batch_size: 32
  gradient_accum_steps: ${eval:'${effective_batch_size} // ${train.batch_size} // ${train.n_gpus}'}
```

> **Note:** GAN training does not currently support `gradient_accum_steps > 1`.

---

## 15. Checkpointing & Resuming

### Automatic Saving

Checkpoints are saved to `save_dir` at a configurable frequency:

```yaml
train:
  save_dir: /path/to/run
  checkpoint_save_frequency: 1000
```

Two files are always maintained:
- `model_latest.pt` — latest checkpoint with optimizer state
- `model_step_N.pt` — step-specific checkpoint (if `save_separate_stepwise_checkpoints: true`)

Each checkpoint stores:
```python
{
    "n_steps_done": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "loss": float,
    # + gan_loss_state_dict and disc_optimizer_state_dict if GAN
}
```

### Save Options

```yaml
train:
  save_separate_stepwise_checkpoints: true   # save model_step_N.pt
  save_optimizer_in_all_checkpoints: true    # include optimizer in every model_step_N.pt
  no_save_weights: false                     # set true to disable all saving
  checkpoint_save_iterations: [500, 1500, 7777]   # force-save model_step_N.pt at these exact steps
```

`checkpoint_save_iterations` writes a stepwise checkpoint at exactly the listed iterations regardless of `checkpoint_save_frequency` or `save_separate_stepwise_checkpoints`. Useful for pinning checkpoints at eval/publication milestones.

### Resume from Latest

```yaml
train:
  resume: true
  save_dir: /path/to/run   # looks for model_latest.pt here
```

If no checkpoint exists, training starts from scratch.

### Resume from Specific Checkpoint

```yaml
train:
  resume_checkpoint_path: /path/to/model_step_5000.pt
  resume_steps_num: 5000   # optional: override step count from checkpoint
```

Also supports HTTP URLs:

```yaml
train:
  resume_checkpoint_path: https://example.com/model_step_5000.pt
```

### Loading Pre-trained Weights (Not Resuming)

To start a new training run from pre-trained weights (resets step counter):

```yaml
model:
  name: my_module.MyModel
  ckpt_path: /path/to/pretrained.pt
```

---

## 16. GAN Training

GAN training requires a special `gan_loss` configuration that wraps a discriminator and provides both generator and discriminator losses.

### GAN Loss Class Interface

Your `GANLoss` class must be an `nn.Module` with two callables available as attributes:

```python
class GANLoss(nn.Module):
    def __init__(self, real_image_key="inp", generated_image_key="generated"):
        super().__init__()
        self.discriminator = Discriminator()

        # Must expose these two attributes:
        self.generator_loss = GeneratorLoss(self.discriminator, ...)
        self.discriminator_loss = DiscriminatorLoss(self.discriminator, ...)
```

- `discriminator_loss(batch, model_outs_detached)` — trains the discriminator
- `generator_loss(batch, model_outs)` — trains the generator

### GAN Config

```yaml
model:
  name: my_gan.Generator

losses:
  dummy:
    weight: 1   # placeholder main loss

gan_loss:
  name: my_gan.GANLoss
  gen_loss_weight: 1.0   # weight for generator adversarial loss
  args:
    real_image_key: inp
    generated_image_key: generated_image

train:
  optimizer:
    name: adam
    args:
      lr: 2e-4
      betas: [0.5, 0.999]

  gan_optimizer:
    name: adam
    args:
      lr: 2e-4
      betas: [0.5, 0.999]
```

### Training Flow

On each step with GAN loss:
1. Generator forward pass → `model_outs`
2. Discriminator loss computed on detached generator output → `disc_optimizer.step()`
3. Main loss + `gen_loss_weight * generator_loss` → `optimizer.step()`
4. Both `gen_loss` and `disc_loss` logged automatically

---

## 17. Crash Detection & Recovery

### Crash Detection

Define thresholds for loss or metric values. If any threshold is breached, a crash is detected:

```yaml
train:
  crash_detect_params:
    total_loss:
      max: 100.0    # trigger if total_loss > 100
    accuracy:
      min: 0.001    # trigger if accuracy < 0.001
```

A NaN in the loss also triggers a crash automatically.

### Recovery Modes

```yaml
train:
  crash_recovery_mode: resume_from_latest_checkpoint
  # Options:
  # "exit"                          — assert False, stops training
  # "resume_from_latest_checkpoint" — reload model_latest.pt and continue
  # "reinit"                        — (raises an error, not fully implemented)
```

Under DDP, crash status is broadcast across all ranks via `all_reduce` so all processes take the same recovery action.

---

## 18. OmegaConf Features

The config is parsed by [OmegaConf](https://omegaconf.readthedocs.io/), giving access to variable interpolation and expression evaluation.

### Variable Interpolation

Reference other config values with `${key}`:

```yaml
n_tokens: 2048
num_layers: 20

model:
  args:
    sequence_len: ${n_tokens}   # resolves to 2048
    n_layer: ${num_layers}      # resolves to 20
```

For nested keys: `${train.batch_size}`.

### Expression Evaluation

Use `${eval:'<python expression>'}` for computed values:

```yaml
n_embd: ${eval:'${num_layers} * 64'}           # 20 * 64 = 1280
n_heads: ${eval:'max(1, (${n_embd} + 127) // 128)'}

effective_batch_size: 256
train:
  gradient_accum_steps: ${eval:'${effective_batch_size} // ${train.batch_size} // ${train.n_gpus}'}
```

The `eval` resolver is registered automatically by the CLI.

### Config Merging

The CLI merges your config on top of the default config, so you only need to specify values that differ from defaults.

---

## 19. GPU Selection

### Explicit List

```yaml
gpus: [0, 2]   # use GPU 0 and GPU 2 only
```

Sets `CUDA_VISIBLE_DEVICES=0,2` before training.

### Auto Selection

```yaml
gpus: auto
train:
  n_gpus: 2
```

Queries GPU memory usage and selects the `n_gpus` least-utilized GPUs (below 90% memory usage threshold).

### Environment Variable

If `gpus` is not set, `CUDA_VISIBLE_DEVICES` is not modified, so whatever is set in the environment is used. You can set it externally:

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py config.yml
```

---

## 20. Asset Downloads

Download datasets or model weights before training begins (master process only, then a barrier for DDP):

### URL Download

```yaml
download_assets:
  my_dataset:
    url: https://example.com/data.tar.gz
```

### Function-Based Download

```yaml
download_assets:
  my_data:
    function: my_module.download_my_dataset
    args:
      destination: /tmp/data
      split: train
```

The function is called as `my_module.download_my_dataset(destination="/tmp/data", split="train")`.

---

## 21. Sanity Check Mode

Run a short end-to-end check before committing to a full training run:

```yaml
sanity: true
```

When `sanity: true`, the framework overrides these settings:
- `n_total_steps = 18`
- `save_dir = /tmp/sanity_check`
- `num_eval_every_steps = 4`
- `num_validation_steps = 10`
- `checkpoint_save_frequency = 6`
- `summary_frequency = 6`
- `training_outs_log_frequency = 5`
- `log_frequency = 3`

The sanity check directory is wiped clean before each run. Use this to verify your model, dataset, and loss are correctly wired up before launching a long training job.

---

## 22. Summary & Logging Functions

### `summary_functions`

Called periodically during training (every `summary_frequency` steps) on the master process. Useful for generating sample outputs, plots, or previews.

**Interface:**

```python
class MySummaryFunction:
    def __init__(self, save_images=True):
        self.save_images = save_images

    def run(self, model, dataset, out_dir, iter):
        """
        model   — the unwrapped model (module, not DDP/DataParallel wrapper)
        dataset — val_dataset if defined, else train dataset
        out_dir — save_dir from config (may be None)
        iter    — current step number
        """
        model.eval()
        with torch.no_grad():
            # generate samples, log images, etc.
            pass
        model.train()
```

**Config:**

```yaml
summary_functions:
  - name: my_module.MySummaryFunction
    args:
      save_images: true
  - name: my_module.AnotherFunction
```

### `training_outs_log_functions`

Called every `training_outs_log_frequency` steps during training, receiving live batch and model outputs:

**Interface:**

```python
class MyOutputLogger:
    def __init__(self):
        pass

    def run(self, model, dataset, out_dir, iter, model_outputs, batch):
        """
        model_outputs — dict returned by model.forward(batch)
        batch         — current training batch dict
        """
        # log images, audio, text, etc.
        pass
```

**Config:**

```yaml
training_outs_log_functions:
  - name: my_module.MyOutputLogger
```

> **Note:** `training_outs_log_functions` are not called when `is_compile_model_with_loss: true`, because model outputs are fused inside the compiled graph.

---

## 23. Tutorials

### Tutorial 1: MNIST Classification

A complete walkthrough of `examples/mnist/`.

**Model** (`examples/mnist/mnist.py`):

```python
class MNISTClassifier(nn.Module):
    def __init__(self, n_hidden=128, random_add_nan=False):
        super().__init__()
        self.random_add_nan = random_add_nan
        self.fc1 = nn.Linear(28 * 28, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x['inp']             # pull image tensor from batch dict
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return {"pred_logits": x}  # return dict with logits
```

**Dataset** (`examples/mnist/mnist.py`):

```python
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(root='/tmp/dataMN', train=train,
                                       download=True, transform=transform)
        self.collate_fn = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inp, gt = self.dataset[idx]
        return {"inp": inp, "gt_class_id": gt}
```

**Config** (`examples/mnist/basic.yml`):

```yaml
model:
  name: examples.mnist.mnist.MNISTClassifier
  args:
    n_hidden: 128
  ckpt_path: null

dataset:
  name: examples.mnist.mnist.MNISTDataset
  args:
    train: true

val_dataset:
  name: examples.mnist.mnist.MNISTDataset
  args:
    train: false

losses:
  cce:
    function_name: cross_entropy_classification_loss
    weight: 1.0
    src_key: pred_logits
    tgt_key: gt_class_id

metrics:
  accuracy:
    function_name: accuracy_classification
    weight: 1.0
    src_key: pred_logits
    tgt_key: gt_class_id

sanity: false

train:
  batch_size: 32
  n_total_steps: 1000
  n_total_samples: -1
  checkpoint_save_frequency: 100
  summary_frequency: 100
  save_dir: /tmp/aa
  device: cuda
  resume_checkpoint_path: null
  resume_steps_num: null
  num_workers: 0
  optimizer:
    name: adam
  num_eval_every_steps: 100
  resume: false
  validation_use_tqdm: false
  no_save_weights: false
  n_gpus: 1
  use_ddp: false
  find_unused_parameters: false
  gradient_accum_steps: 2
  use_tqdm: true
  grad_clip_value: 0
  use_bfloat16_autocast: true
  compile_model: true
  is_compile_model_with_loss: false
  pt_single_threaded: true
  save_separate_stepwise_checkpoints: true
```

**Run:**

```bash
python train.py examples/mnist/basic.yml
```

**Key observations:**
- The batch from `MNISTDataset` has keys `inp` and `gt_class_id`
- The model reads `batch["inp"]` and returns `{"pred_logits": logits}`
- The loss config links `src_key: pred_logits` → model output, `tgt_key: gt_class_id` → batch
- bfloat16 autocast and model compilation are enabled for speed
- The `metrics` section computes accuracy on both train and validation without affecting the loss

---

### Tutorial 2: Multi-GPU DDP Training

Scale the MNIST example to 2 GPUs using DDP. Only the `train` section changes:

**Config** (`examples/mnist/basic_ddp.yml`):

```yaml
model:
  name: examples.mnist.mnist.MNISTClassifier
  args:
    n_hidden: 128
  ckpt_path: null

dataset:
  name: examples.mnist.mnist.MNISTDataset
  args:
    train: true

val_dataset:
  name: examples.mnist.mnist.MNISTDataset
  args:
    train: false

losses:
  cce:
    function_name: cross_entropy_classification_loss
    weight: 1.0
    src_key: pred_logits
    tgt_key: gt_class_id

train:
  n_gpus: 2
  use_ddp: true
  save_dir: /tmp/aa
  device: cuda
  optimizer:
    name: adam
```

**Run** (identical command — torchrun is automatic):

```bash
python train.py examples/mnist/basic_ddp.yml
```

train_loop detects that DDP is requested but `LOCAL_RANK` is not set, and re-executes the process as:

```
torchrun --nproc_per_node=2 --master_port=<random> train.py examples/mnist/basic_ddp.yml
```

**What happens under DDP:**
- tqdm is disabled automatically (avoids duplicate output)
- Dataset is loaded on master first, then workers (prevents download races)
- A `DistributedSampler` is used for the training dataloader
- Validation runs on master only
- Gradients are synced every `gradient_accum_steps` steps (using `model.no_sync()` for inner steps)
- Checkpoints are saved by master only

---

### Tutorial 3: GAN Training

A walkthrough of `examples/simple_gan/`.

**Generator** (returns generated images):

```python
class Generator(nn.Module):
    def forward(self, batch):
        device = batch['inp'].device
        z = torch.randn(batch['inp'].size(0), 100, 1, 1, device=device)
        x = self.decode(z)
        return {"generated_image": x}
```

**GANLoss** (wraps discriminator + both loss functions):

```python
class GANLoss(nn.Module):
    def __init__(self, real_image_key="inp", generated_image_key="generated_image"):
        super().__init__()
        self.discriminator = Discriminator()
        self.generator_loss = GeneratorLoss(self.discriminator, ...)
        self.discriminator_loss = DiscriminatorLoss(self.discriminator, ...)
```

**Config** (`examples/simple_gan/basic.yml`):

```yaml
model:
  name: examples.simple_gan.simple_gan.Generator

dataset:
  name: examples.simple_gan.simple_gan.MNISTDataset
  args:
    train: true

val_dataset:
  name: examples.simple_gan.simple_gan.MNISTDataset
  args:
    train: false

losses:
  dummy:
    weight: 1

training_outs_log_functions:
  - name: examples.simple_gan.simple_gan.LogTrainOutputs

gan_loss:
  name: examples.simple_gan.simple_gan.GANLoss
  gen_loss_weight: 1
  args:
    real_image_key: inp
    generated_image_key: generated_image

sanity: false

train:
  batch_size: 512
  n_total_steps: 1000
  n_total_samples: -1
  checkpoint_save_frequency: 100
  summary_frequency: 100
  training_outs_log_frequency: 100
  save_dir: /tmp/aa
  device: cuda
  optimizer:
    name: adam
    args:
      lr: 2e-4
      betas: [0.5, 0.999]
  gan_optimizer:
    name: adam
    args:
      lr: 2e-4
      betas: [0.5, 0.999]
```

**Key points:**
- The main `losses` uses `dummy` (returns 0) because the generator is trained solely via `gan_loss`
- `gan_loss.gen_loss_weight: 1` scales the adversarial generator loss
- `training_outs_log_functions` logs ASCII art of real vs. generated images every 100 steps
- Both generator and discriminator use the same Adam hyperparameters (β₁=0.5 is standard for GANs)

---

### Tutorial 4: LLM Training (NanoChat)

Advanced configuration demonstrating: multi-optimizer, custom LR scheduler, OmegaConf expressions, model compilation, and meta-device init.

**Config** (`examples/nanochat/nanochat.yml`):

```yaml
# Computed top-level variables (used via interpolation below)
n_tokens: 2048
num_layers: 20
n_embd: ${eval:'${num_layers} * 64'}          # 1280
n_heads: ${eval:'max(1, (${n_embd} + 127) // 128)'}  # 10

model:
  name: examples.nanochat.gpt.GPT
  args:
    config:
      sequence_len: ${n_tokens}
      vocab_size: 65536
      n_layer: ${num_layers}
      n_head: ${n_heads}
      n_kv_head: ${n_heads}
      n_embd: ${n_embd}
  ckpt_path: null

orig_nanochat_path: /home/ubuntu/nanochat

dataset:
  name: examples.nanochat.dataset.BasicTokensDL
  args:
    parquet_data_dir: ${orig_nanochat_path}/data/base_data
    tokenizer_dir: ${orig_nanochat_path}/data/tokenizer
    n_tokens: ${n_tokens}
    split: train

val_dataset:
  name: examples.nanochat.dataset.BasicTokensDL
  args:
    parquet_data_dir: ${orig_nanochat_path}/data/base_data
    tokenizer_dir: ${orig_nanochat_path}/data/tokenizer
    n_tokens: ${n_tokens}
    split: val
    len: 400

losses:
  cce:
    function_name: examples.nanochat.gpt.llm_cce_loss
    weight: 1.0
    src_key: out_logits
    tgt_key: gt_output_ids

sanity: false

# LR scale variables
embedding_lr: 0.2
unembedding_lr: 0.004
dmodel_lr_scale: ${eval:'(${n_embd} / 768) ** -0.5'}

effective_batch_size: 256

train:
  batch_size: 32
  gradient_accum_steps: ${eval:'${effective_batch_size} // ${train.batch_size} // ${train.n_gpus}'}
  n_total_steps: 21500
  num_eval_every_steps: 1000
  checkpoint_save_frequency: 1000
  summary_frequency: 1000
  save_dir: /tmp/aa
  device: cuda
  resume_checkpoint_path: null
  num_workers: 0
  resume: false
  n_gpus: 1
  use_ddp: false
  find_unused_parameters: false
  compile_model: true
  use_bfloat16_autocast: true
  is_compile_model_with_loss: true   # fuse model + loss for max throughput
  create_model_meta_init: true       # init on meta device for large model
  grad_clip_value: 1.0

  optimizer:
    name: multi                       # two separate optimizers
    optimizers:
      embeds:
        name: adamw
        args:
          weight_decay: 0.0
          betas: [0.8, 0.95]
          eps: 1e-10
        param_groups:
          - params: ["transformer.wte"]
            lr: ${eval:'${embedding_lr} * ${dmodel_lr_scale}'}
          - params: ["lm_head"]
            lr: ${eval:'${unembedding_lr} * ${dmodel_lr_scale}'}
      matrix:
        name: examples.nanochat.muon_opt.Muon
        args:
          lr: 0.02
          momentum: 0.95
        param_groups:
          - params: ["transformer.h"]
            lr: 0.02

  lr_scheduler:
    name: examples.nanochat.lr_scheduler.AKLRScheduler
    args:
      warmup_ratio: 0.0
      warmdown_ratio: 0.2
      final_lr_frac: 0.0
      muon_optimizer_name: matrix
      total_iterations: ${train.n_total_steps}
```

**Key design decisions:**
- `create_model_meta_init: true` — for a 1B+ parameter model, avoids allocating full model on CPU before GPU transfer
- `is_compile_model_with_loss: true` — fuses the language model forward + cross-entropy into a single compiled kernel
- Multi-optimizer assigns different LR scaling to embedding/unembedding layers vs. transformer blocks (μP-inspired)
- `gradient_accum_steps` is derived automatically from `effective_batch_size` using OmegaConf eval
- The custom Muon optimizer is loaded via dotpath without any framework changes

---

### Tutorial 5: Fine-tuning Stable Diffusion with LoRA

Demonstrates: summary functions for image previews, `optimizer_metrics`, cosine LR scheduler, and parameter groups for LoRA.

**Config** (`examples/stable_diffusion_lora/basic.yml`):

```yaml
model:
  name: examples.stable_diffusion_lora.model.StableDiffusionModel
  args:
    pretrained_model_name_or_path: "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_rank: 256
  ckpt_path: null

dataset:
  name: examples.stable_diffusion_lora.model.TextImageDataset
  args:
    dataset_name: "Norod78/microsoft-fluentui-emoji-512-whitebg"
    text_prepend: "sks style "

losses:
  mse:
    weight: 1.0
    src_key: predicted_noise
    tgt_key: target_noise

summary_functions:
  - name: examples.stable_diffusion_lora.model.GenerateBasicPreview
    args:
      show_imgcat: true
      prompts:
        - "sks style taj mahal"
        - "sks style a futuristic cityscape at sunset"
        - "sks style a beautiful landscape with mountains and a river"
      img_size: 512

optimizer_metrics:
  lr:
    function_name: opt_learning_rate   # logs current LR each step

sanity: false

train:
  batch_size: 4
  gradient_accum_steps: 4
  n_total_steps: 10000
  n_total_samples: -1
  summary_frequency: 20
  checkpoint_save_frequency: 100
  grad_clip_value: 1.0
  save_dir: null
  device: cuda

  optimizer:
    name: adamw
    args:
      lr: 1e-4
      weight_decay: 1e-2
    param_groups:
      - params: ["lora_params"]   # only train LoRA parameters
        lr: 1e-4

  lr_scheduler:
    name: train_loop.lr_schedulers.CosineLRScheduler
    args:
      total_steps: ${train.n_total_steps}
```

**Key design decisions:**
- `summary_functions` runs every 20 steps, generating preview images with 3 different prompts
- `optimizer_metrics.lr` logs the current learning rate to `train_loss.jsonl` each step
- `param_groups` restricts training to only the LoRA parameters (the base model is frozen)
- The cosine LR scheduler decays from the optimizer's initial LR to 0 over `n_total_steps`
- `save_dir: null` skips checkpointing (useful during experimentation)

---

## 24. Python API

Use these snippets to call train_loop utilities directly from Python — for inference, evaluation, or data inspection outside a training run.

### 1. Loading a Config

```python
from omegaconf import OmegaConf
from train_loop.default_config import DEFAULT_CONFIG

config = OmegaConf.load("config.yml")
config = OmegaConf.merge(DEFAULT_CONFIG, config)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.resolve(config)
```

### 2. Building a Dataset & DataLoader from Config

```python
from train_loop.utils.dynamic_import import build_class
from torch.utils.data import DataLoader

dataset = build_class(config.dataset)
collate_fn = getattr(dataset, "collate_fn", None)
dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                        collate_fn=collate_fn, num_workers=config.train.num_workers,
                        shuffle=True)
```

### 3. Load Latest Trained Model (one-liner)

```python
from train_loop.utils.inference_utils import load_model_from_config_path

# accepts: path to yaml, path to save_dir (contains config.yaml), or OmegaConf object
model = load_model_from_config_path("path/to/config.yml")
# or: model = load_model_from_config_path("/tmp/my_run")  # finds config.yaml inside
model.eval()
```

### 4. Manual Model Build + Checkpoint Load

```python
from train_loop.utils.dynamic_import import build_class
from train_loop.utils.model_loading import get_latest_checkpoint
import torch

model = build_class(config.model)
model.to("cuda")

ckpt_path, step = get_latest_checkpoint(config.train.save_dir)
if ckpt_path:
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from step {step}")
model.eval()
```

### 5. Download a File (with caching)

```python
from train_loop.utils.download import download_file

local_path = download_file("https://example.com/weights.pt")
# Returns path as-is for non-http strings (already local)
local_path = download_file("/already/local/file.pt")  # no-op, returns same path
```

### 6. Running Inference on a Batch

```python
from train_loop.utils.model_utils import move_to_device
import torch

device = "cuda"
model.eval()
with torch.no_grad():
    batch = move_to_device(batch, device)
    outputs = model(batch)  # outputs is a dict
```

### 7. Inspect Model Parameter Counts

```python
from train_loop.utils.model_utils import print_params_count_mb

print_params_count_mb(model, depth=2)
# prints per-submodule parameter sizes in MB
```

---

## 25. CLI Reference

### Basic Usage

```bash
python train.py <config.yml> [overrides...]
```

Or using the module form (compatible with torchrun auto-restart):

```bash
python -m train_loop.train <config.yml> [overrides...]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to a YAML config file. Pass `none` or `null` to use only defaults. |
| `overrides` | Zero or more `key=value` pairs in dotpath notation |

### Override Syntax

Overrides use OmegaConf dotpath notation and are applied after loading the YAML file:

```bash
# Override a train hyperparameter
python train.py config.yml train.batch_size=64

# Override model args
python train.py config.yml model.args.hidden=512

# Override multiple values
python train.py config.yml train.batch_size=64 train.n_total_steps=10000 train.save_dir=/tmp/run2

# Change device
python train.py config.yml train.device=cpu

# Enable DDP
python train.py config.yml train.use_ddp=true train.n_gpus=4

# Override nested optimizer args
python train.py config.yml train.optimizer.args.lr=0.001
```

Override keys are validated against the loaded config by default. An error is raised if you try to set a key that doesn't exist in the config. To disable validation:

```bash
python train.py config.yml validate_override_keys=false train.new_key=value
```

### Output Files

All output files are written to `train.save_dir`:

| File | Description |
|------|-------------|
| `config.yaml` | Resolved config (with all interpolations expanded) |
| `model_latest.pt` | Latest checkpoint (model + optimizer state) |
| `model_step_N.pt` | Step-specific checkpoint (if `save_separate_stepwise_checkpoints: true`) |
| `train_loss.jsonl` | Per-step training losses and metrics (one JSON object per line) |
| `val_loss.jsonl` | Per-validation-run losses and metrics |
| `git_state.txt` | Git commit hash and diff for reproducibility |

### Reading Loss Logs

```python
import json

with open("/tmp/run/train_loss.jsonl") as f:
    steps = [json.loads(line) for line in f]

# steps[i] = {"step": 100, "total_loss": 1.23, "cce": 1.23, "accuracy": 0.45, ...}
```

---

## 26. Weights & Biases Integration

train_loop has built-in [Weights & Biases](https://wandb.ai) support. When enabled, every training metric and validation metric is logged automatically — no extra code required.

### Setup

Install the client:

```bash
pip install wandb
```

Authenticate via environment variable (recommended):

```bash
export WANDB_API_KEY=your_key_here
```

Or pass the key directly in the config (see below).

### Configuration

Add a `wandb:` block to your YAML config:

```yaml
wandb:
  project: my_project       # W&B project name (required)
  name: my_run_name         # run display name (optional, auto-generated if omitted)
  entity: my_team           # W&B entity / team (optional)
  api_key: your_key_here    # optional — prefer WANDB_API_KEY env var instead
  tags:                     # optional list of tags
    - experiment_1
    - baseline
```

If `wandb:` is omitted or set to `null`, W&B logging is skipped entirely.

### What Gets Logged

| W&B key prefix | Source |
|----------------|--------|
| `train/*` | All train losses, metrics, optimizer metrics, grad metrics — logged every `log_frequency` steps |
| `val/*`   | All validation losses and metrics — logged at every `num_eval_every_steps` |

The full resolved config is also uploaded to the run as W&B config.

### Example

```yaml
train:
  batch_size: 16
  n_total_steps: 100000
  save_dir: /tmp/my_run

wandb:
  project: audio_tts
  name: soprano_film_arch
  entity: my_org
```

```bash
WANDB_API_KEY=xxx python -m train_loop.train configs/my_config.yml
```

---

## 27. Dataset Demo

`train_loop.dataset_demo` is an interactive Gradio browser for any dataset config. It lets you inspect individual samples — viewing tensors as audio players or images, and all other values as text — without writing any extra code.

### Usage

```bash
python -m train_loop.dataset_demo configs/my_config.yml
python -m train_loop.dataset_demo configs/my_config.yml --port 7861 --share
```

The YAML must contain a top-level `dataset:` key (same format used by `train_loop.train`).

### Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to YAML config with a `dataset:` key |
| `--port` | Gradio server port (default: 7860) |
| `--share` | Create a public Gradio share link |

### Features

- **Index slider** — scrub through dataset samples; press **Load** to fetch the selected index
- **Auto-detection** — tensor fields are heuristically identified as audio (1-D float arrays > 1000 samples, or keys containing `audio`/`wav`/`waveform`) or images (CHW tensors, or keys containing `image`/`mel`/`spec`)
- **Configurable rendering** — override which keys are treated as audio or image using the comma-separated text boxes; changes take effect on the next **Load**
- **Sample rate control** — set the playback sample rate for audio tensors
- **Text fallback** — non-tensor values (strings, dicts, lists) are shown as formatted text; tensors that aren't audio/image show shape/dtype/min/max/mean

### Example

```yaml
# configs/my_audio_dataset.yml
dataset:
  name: mydatasets.audio_dataset_v2.AudioDatasetV2
  args:
    src_dataset:
      name: mydatasets.arrow_dataset.JsonDataset
      args:
        json_file: assets/my_data.jsonl
        file_keys_to_load: [audio]
    audio_key: audio
    target_sample_rate: [16000, 24000]
```

```bash
python -m train_loop.dataset_demo configs/my_audio_dataset.yml --share
```
