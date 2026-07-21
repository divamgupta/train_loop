"""
Multi-parquet streamer — yields rows from multiple parquet files sequentially.

Wraps stream_parquet_rows to iterate over many parquet files. Supports three
input modes:

  1. Explicit list of parquet keys
  2. A generator/iterator that yields parquet keys
  3. S3/R2 bucket listing with optional prefix and contains filter

Usage:

    from train_loop.datasets.multi_parquet_streamer import multi_stream_parquet
    from train_loop.datasets.stream_parquet import CloudStorageConfig

    cfg = CloudStorageConfig(bucket="granary", endpoint_url="...",
                             access_key_id="...", secret_access_key="...")

    # 1. From a list of keys
    for row in multi_stream_parquet(cfg, keys=["path/a.parquet", "path/b.parquet"]):
        print(row)

    # 2. From a generator
    def my_gen():
        for i in range(100):
            yield f"chunked_parquet/data_chunk_{i:06d}.parquet"
    for row in multi_stream_parquet(cfg, keys_generator=my_gen()):
        print(row)

    # 3. From bucket listing (list all parquets under a prefix)
    for row in multi_stream_parquet(cfg, prefix="chunked_parquet/en_yodas/"):
        print(row)

    # 4. Bucket listing with contains filter
    for row in multi_stream_parquet(cfg, prefix="chunked_parquet/", contains="en_yodas"):
        print(row)

    # All modes support shuffle, loop, and all stream_parquet_rows kwargs.
"""

import logging
import random
from typing import Iterator, List, Optional, Union

from .stream_parquet import CloudStorageConfig, make_client, stream_parquet_rows

log = logging.getLogger("multi_parquet_streamer")


def _list_parquet_keys(s3, bucket, prefix, contains=None, suffix=".parquet"):
    """List all parquet keys under a prefix in the bucket."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if suffix and not key.endswith(suffix):
                continue
            if contains and contains not in key:
                continue
            keys.append(key)
    log.info("Listed %d parquet files under %s (contains=%s)", len(keys), prefix, contains)
    return keys


def multi_stream_parquet(
    config: CloudStorageConfig,
    keys: Optional[List[str]] = None,
    keys_generator: Optional[Iterator[str]] = None,
    prefix: Optional[str] = None,
    contains: Optional[str] = None,
    shuffle: bool = False,
    loop: bool = False,
    seed: Optional[int] = None,
    # Passed through to stream_parquet_rows:
    decode_audio: bool = True,
    target_sampling_rate: Optional[int] = None,
    batch_size: int = 64,
    columns: Optional[List[str]] = None,
    bulk: bool = False,
    max_retries: Optional[int] = None,
):
    """
    Yield rows from multiple parquet files, one after another.

    Exactly one source must be provided:
      keys:            explicit list of S3/R2 object keys
      keys_generator:  an iterator/generator that yields keys on demand
      prefix:          list all .parquet files under this S3 prefix

    Args:
      config:        CloudStorageConfig for the bucket.
      keys:          list of parquet keys to stream from.
      keys_generator: iterator that yields parquet keys.
      prefix:        S3 prefix to list parquet files from.
      contains:      filter listed keys to those containing this substring.
      shuffle:       shuffle the key order (only for keys/prefix mode, not generator).
      loop:          loop forever over the files (for training).
      seed:          random seed for shuffle reproducibility.
      decode_audio:  decode audio columns (default True).
      target_sampling_rate: resample audio to this rate.
      batch_size:    pyarrow batch size per row-group read.
      columns:       subset of columns to read.
      bulk:          download whole file at once (vs row-group streaming).
      max_retries:   None = retry forever on network errors.

    Yields:
      dict per row, same as stream_parquet_rows.
    """
    # Validate inputs
    sources = sum(x is not None for x in [keys, keys_generator, prefix])
    if sources != 1:
        raise ValueError("Provide exactly one of: keys, keys_generator, prefix")

    s3 = make_client(config)

    # Build the key source
    if prefix is not None:
        keys = _list_parquet_keys(s3, config.bucket, prefix, contains=contains)
        if not keys:
            raise ValueError(f"No parquet files found under prefix={prefix!r} "
                             f"(contains={contains!r}) in bucket={config.bucket!r}")
        log.info("Found %d parquet files", len(keys))

    # Stream kwargs passed through to stream_parquet_rows
    stream_kwargs = dict(
        decode_audio=decode_audio,
        target_sampling_rate=target_sampling_rate,
        batch_size=batch_size,
        columns=columns,
        bulk=bulk,
        max_retries=max_retries,
    )

    if keys_generator is not None:
        # Generator mode: can't shuffle or know total count upfront
        if loop:
            raise ValueError("loop=True is not supported with keys_generator "
                             "(generator can only be consumed once)")
        for key in keys_generator:
            log.info("Streaming from %s", key)
            yield from stream_parquet_rows(config, key, s3=s3, **stream_kwargs)
        return

    # List mode (keys from explicit list or prefix listing)
    rng = random.Random(seed)
    iteration = 0

    while True:
        iteration += 1
        order = list(keys)
        if shuffle:
            rng.shuffle(order)

        for key in order:
            log.info("Streaming from %s (iteration %d)", key, iteration)
            yield from stream_parquet_rows(config, key, s3=s3, **stream_kwargs)

        if not loop:
            break


if __name__ == "__main__":
    import os, sys, time

    cfg = CloudStorageConfig(
        bucket=os.environ["CLOUD_STORAGE_BUCKET"],
        endpoint_url=os.environ["CLOUD_STORAGE_ENDPOINT_URL"],
        access_key_id=os.environ["CLOUD_STORAGE_ACCESS_KEY_ID"],
        secret_access_key=os.environ["CLOUD_STORAGE_SECRET_ACCESS_KEY"],
    )

    prefix = sys.argv[1] if len(sys.argv) > 1 else None
    if not prefix:
        print("Usage: python -m train_loop.datasets.multi_parquet_streamer <prefix> [contains] [max_rows]")
        sys.exit(1)
    contains = sys.argv[2] if len(sys.argv) > 2 else None
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n = 0
    t0 = time.time()
    for row in multi_stream_parquet(cfg, prefix=prefix, contains=contains, shuffle=True, seed=42):
        n += 1
        a = row.get("audio", {})
        sr = a.get("sampling_rate", 0)
        dur = len(a.get("array", [])) / sr if sr else 0
        print(f"[{n}] {dur:.1f}s {sr}Hz | {row.get('text', '')[:60]!r}")
        if n >= max_rows:
            break
    dt = time.time() - t0
    print(f"\nStreamed {n} rows in {dt:.1f}s ({n / dt:.1f} rows/s)")
