import argparse
import os
import sys
from omegaconf import OmegaConf
import torch

from .train_module import train
from .default_config import DEFAULT_CONFIG
from .utils.gpu_utils import get_free_gpus_ids
from .utils.device_utils import set_visible_devices

def train_cli():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--config2', type=str, default=None,
                        help='Optional 2nd YAML that overrides keys in the main config (applied after main YAML, before CLI overrides)')
    parser.add_argument('overrides', nargs='*',
                        help='Override config values using dotpath notation (e.g., train.lr=0.001, model.hidden_size=256)')
    args = parser.parse_args()
    OmegaConf.register_new_resolver("eval", eval)
    if args.config.lower() == 'none' or args.config is None or args.config.lower() == 'null':
        config = DEFAULT_CONFIG
    else:
        config = OmegaConf.load(args.config)
        config = OmegaConf.merge(DEFAULT_CONFIG, config)

    # Apply --config2 YAML overrides (keys must exist in the merged config)
    if args.config2:
        config2 = OmegaConf.load(args.config2)
        config = OmegaConf.merge(config, config2)
        print(f"Applied --config2: {args.config2}")
    
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
        OmegaConf.resolve(config)
        print(f"Applied overrides: {args.overrides}")

    if 'gpus' in config and config.gpus is not None and config.gpus != "" and config.gpus != []:
        if isinstance(config.gpus, str) and config.gpus.lower() == 'auto':
            config.gpus = get_free_gpus_ids(num=config.train.n_gpus, threshold=0.9)
            print(f"Auto-selected free GPU IDs: {config.gpus}")
        if isinstance(config.gpus, int):
            config.gpus = [config.gpus]
        if isinstance(config.gpus, str):
            config.gpus = [int(gpu_id.strip()) for gpu_id in config.gpus.split(',')]
        gpu_ids = [int(g) for g in config.gpus]
        # Sets CUDA_VISIBLE_DEVICES + HIP_VISIBLE_DEVICES (AMD). The HIP var is
        # inert on NVIDIA/CPU, so this is safe regardless of train.rocm.
        set_visible_devices(gpu_ids)

    # --- DDP torchrun auto-restart logic ---
    use_ddp = config.train.get('use_ddp', False)
    n_gpus = config.train.get('n_gpus', 1)
    if n_gpus == 'auto':
        n_gpus = torch.cuda.device_count()
    local_rank_env = os.environ.get('LOCAL_RANK', None)
    # If DDP is enabled, but not run with torchrun (LOCAL_RANK not set), restart with torchrun
    use_accelerate = config.train.get('use_accelerate', False)
    if use_ddp and local_rank_env is None:
        print("DDP is enabled but not running under torchrun. Restarting with torchrun...")

        random_port = 29500 + int.from_bytes(os.urandom(2), "big") % 1000

        # Base torchrun command
        if not use_accelerate:
            # Use sys.executable so the same interpreter (e.g. a conda env) is
            # re-launched; a bare "python" may not be on PATH.
            torchrun_cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={n_gpus}",
                f"--master_port={random_port}",
            ]
        else:
            torchrun_cmd = [
                "accelerate",
                "launch",
                f"--multi_gpu",
            ]

        # If the script was started with `python -m <module>`, prefer restarting
        # using: torchrun ... python -m <module> <config> ...
        module_name = None
        spec = globals().get("__spec__", None)
        if spec and getattr(spec, "name", None):
            module_name = spec.name

        if module_name:
            # Use the same python executable and module form
            torchrun_cmd.extend([ "-m", module_name, args.config])
        else:
            # Fallback to script path (sys.argv[0]) for normal python script execution
            torchrun_cmd.extend([sys.argv[0], args.config])

        # Forward --config2 if present
        if args.config2:
            torchrun_cmd.extend(["--config2", args.config2])

        # Append any overrides
        torchrun_cmd.extend(args.overrides or [])
        print("Running:", " ".join(torchrun_cmd))
        os.execvp(torchrun_cmd[0], torchrun_cmd)
        return  # Should not reach here

    train(config)

if __name__ == "__main__":
    train_cli()
