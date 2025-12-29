import argparse
import os
import sys
from omegaconf import OmegaConf
import torch

from .train_module import train
from .default_config import DEFAULT_CONFIG

def train_cli():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('overrides', nargs='*',
                        help='Override config values using dotpath notation (e.g., train.lr=0.001, model.hidden_size=256)')
    args = parser.parse_args()
    OmegaConf.register_new_resolver("eval", eval)
    if args.config.lower() == 'none' or args.config is None or args.config.lower() == 'null':
        config = DEFAULT_CONFIG
    else:
        config = OmegaConf.load(args.config)
        config = OmegaConf.merge(DEFAULT_CONFIG, config)
    
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
        if isinstance(config.gpus, int):
            config.gpus = [config.gpus]
        if isinstance(config.gpus, str):
            config.gpus = [int(gpu_id.strip()) for gpu_id in config.gpus.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in config.gpus])

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
            torchrun_cmd = [
                "torchrun",
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

        # Append any overrides
        torchrun_cmd.extend(args.overrides or [])
        print("Running:", " ".join(torchrun_cmd))
        os.execvp(torchrun_cmd[0], torchrun_cmd)
        return  # Should not reach here

    train(config)

if __name__ == "__main__":
    train_cli()
