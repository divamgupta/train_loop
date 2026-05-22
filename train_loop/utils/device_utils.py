"""Device helpers shared by NVIDIA (CUDA) and AMD (ROCm) backends.

PyTorch built against ROCm maps AMD GPUs onto the ``torch.cuda`` API, so a
device string stays a plain ``"cuda"`` / ``"cuda:N"`` on both vendors. The only
AMD-specific behaviour lives here: extra visibility env vars and a couple of
small "auto" resolvers that pick sensible defaults from ``train.device``.
"""

import os
import torch


def set_visible_devices(gpu_ids) -> None:
    """Set GPU visibility env vars for both NVIDIA and AMD toolchains.

    We set ``CUDA_VISIBLE_DEVICES`` (NVIDIA, also honoured by ROCm PyTorch) and
    ``HIP_VISIBLE_DEVICES`` (AMD). The ``HIP_*`` var is inert on NVIDIA systems.

    We deliberately do NOT set ``ROCR_VISIBLE_DEVICES``: it filters devices at a
    lower level than ``HIP_VISIBLE_DEVICES``, so setting both makes HIP re-index
    into the already-ROCR-filtered set (e.g. selecting GPU 2 yields *zero*
    visible devices). ``CUDA_*`` and ``HIP_*`` act at the same level and compose
    safely.
    """
    ids_str = ",".join(str(i) for i in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ids_str
    # AMD/ROCm equivalent (inert on NVIDIA systems)
    os.environ["HIP_VISIBLE_DEVICES"] = ids_str


def validate_device(device: str, rocm: bool = False) -> None:
    """Raise a clear error if the configured device is not available."""
    d = str(device).lower()
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            backend = "ROCm" if rocm else "CUDA"
            raise RuntimeError(
                f"train.device is '{device}' but no {backend} GPU was found. "
                f"Check your drivers/toolkit installation, "
                f"or set train.device to 'cpu'."
            )
        if ":" in d:
            idx = int(d.split(":")[1])
            n = torch.cuda.device_count()
            if idx >= n:
                raise RuntimeError(
                    f"train.device is '{device}' but only {n} GPU(s) are "
                    f"visible. Check your CUDA/HIP_VISIBLE_DEVICES settings."
                )


def get_ddp_backend(requested: str = "auto", device: str = "cuda") -> str:
    """Resolve the DDP backend string.

    'auto' -> 'nccl' for cuda/rocm (AMD ships RCCL, same wire protocol),
              'gloo' for cpu.
    """
    if requested != "auto":
        return requested
    return "gloo" if str(device).startswith("cpu") else "nccl"


def get_autocast_device_type(device: str) -> str:
    """Return the device_type string for torch.amp.autocast().

    Both NVIDIA CUDA and AMD ROCm use 'cuda' in PyTorch >= 2.0.
    """
    return "cpu" if str(device).startswith("cpu") else "cuda"
