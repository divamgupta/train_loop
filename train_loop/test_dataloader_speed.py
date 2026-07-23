"""
Test dataloader speed using the same config as training.

Usage:
    python -m train_loop.test_dataloader_speed configs/my_config.yaml
    python -m train_loop.test_dataloader_speed configs/my_config.yaml train.num_workers=4
"""

import argparse
import time
from collections import deque

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .default_config import DEFAULT_CONFIG
from .utils.dynamic_import import build_class


def main():
    parser = argparse.ArgumentParser(description="Test dataloader speed")
    parser.add_argument("config", type=str, help="Path to training config YAML")
    parser.add_argument("overrides", nargs="*",
                        help="Override config values (e.g. train.num_workers=4)")
    args = parser.parse_args()

    OmegaConf.register_new_resolver("eval", eval)
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(DEFAULT_CONFIG, config)
    if args.overrides:
        override_config = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, override_config)
    OmegaConf.resolve(config)

    batch_size = config.train.batch_size
    num_workers = config.train.num_workers

    print(f"Config: {args.config}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print()

    # Build dataset
    print("Building dataset...")
    t0 = time.time()
    dataset = build_class(config.dataset)
    print(f"Dataset built in {time.time() - t0:.2f}s")
    print(f"Dataset length: {len(dataset)}")

    collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )

    # Warmup
    print("\nWarming up (1 batch)...")
    t0 = time.time()
    batch_iter = iter(dataloader)
    batch = next(batch_iter)
    print(f"Warmup: {time.time() - t0:.2f}s")

    # Print batch info
    if isinstance(batch, dict):
        print("Batch keys:", list(batch.keys()))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print(f"\n{'Batch':<8} {'Current':>10} {'Last-50 avg':>12} {'Total avg':>12} {'Samples/s':>10}")
    print("-" * 56)

    all_times = []
    recent_times = deque(maxlen=50)
    i = 0

    try:
        while True:
            t0 = time.time()
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
            elapsed = time.time() - t0

            all_times.append(elapsed)
            recent_times.append(elapsed)
            i += 1

            total_avg = sum(all_times) / len(all_times)
            recent_avg = sum(recent_times) / len(recent_times)
            sps = batch_size / recent_avg

            print(f"{i:<8} {elapsed*1000:>8.1f}ms {recent_avg*1000:>10.1f}ms {total_avg*1000:>10.1f}ms {sps:>8.1f}/s")
    except KeyboardInterrupt:
        total_avg = sum(all_times) / len(all_times)
        recent_avg = sum(recent_times) / len(recent_times)
        print(f"\n{'='*56}")
        print(f"  Stopped after {i} batches")
        print(f"  Total avg:    {total_avg*1000:.1f}ms/batch")
        print(f"  Last-50 avg:  {recent_avg*1000:.1f}ms/batch")
        print(f"  Throughput:   {batch_size / recent_avg:.1f} samples/sec")
        print(f"{'='*56}")


if __name__ == "__main__":
    main()
