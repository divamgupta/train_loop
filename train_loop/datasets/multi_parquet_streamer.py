"""
Multi-parquet streamer — yields rows from multiple parquet files.

Wraps stream_parquet_rows to iterate over many parquet files. Supports:

  1. Explicit list of parquet keys
  2. A generator/iterator that yields parquet keys
  3. S3/R2 bucket listing with optional prefix and contains filter
  4. Weighted sampling by keyword — assign weights to keywords and files
     matching those keywords are sampled proportionally.

Usage:

    from train_loop.datasets.multi_parquet_streamer import multi_stream_parquet
    from train_loop.datasets.stream_parquet import CloudStorageConfig

    cfg = CloudStorageConfig(bucket="granary", endpoint_url="...",
                             access_key_id="...", secret_access_key="...")

    # From a list of keys
    for row in multi_stream_parquet(cfg, keys=["path/a.parquet", "path/b.parquet"]):
        print(row)

    # From bucket listing with prefix
    for row in multi_stream_parquet(cfg, prefix="chunked_parquet/en_yodas/"):
        print(row)

    # Weighted sampling by keyword (e.g. 3x more "yodas" than "librilight")
    weights = {"yodas": 3.0, "librilight": 1.0, "voxpopuli": 0.5}
    for row in multi_stream_parquet(cfg, prefix="chunked_parquet/", keyword_weights=weights):
        print(row)
"""

import logging
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Union

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


def _assign_weights(keys, keyword_weights):
    """Assign a sampling weight to each key based on keyword matches.

    A key's weight is the weight of the FIRST matching keyword found in its path.
    Keys that match no keyword get weight 0 and are excluded.

    Args:
        keys: list of parquet file paths.
        keyword_weights: {"keyword": weight, ...}

    Returns:
        (filtered_keys, weights) — parallel lists, only keys with weight > 0.
    """
    filtered_keys = []
    weights = []
    unmatched = []

    # Sort keywords longest-first so more specific keywords match before shorter ones
    # e.g. "non_en_yodas" matches before "en_yodas"
    sorted_keywords = sorted(keyword_weights.items(), key=lambda kv: -len(kv[0]))

    for key in keys:
        matched = False
        for keyword, w in sorted_keywords:
            if keyword in key:
                filtered_keys.append(key)
                weights.append(w)
                matched = True
                break
        if not matched:
            unmatched.append(key)

    # Log summary
    by_keyword = defaultdict(int)
    for key, w in zip(filtered_keys, weights):
        for keyword, _ in sorted_keywords:
            if keyword in key:
                by_keyword[keyword] += 1
                break

    for keyword, count in sorted(by_keyword.items()):
        log.info("  keyword %r: %d files, weight=%.2f", keyword, count, keyword_weights[keyword])
    if unmatched:
        log.info("  %d files matched no keyword (excluded)", len(unmatched))

    return filtered_keys, weights


def multi_stream_parquet(
    config: CloudStorageConfig,
    keys: Optional[List[str]] = None,
    keys_generator: Optional[Iterator[str]] = None,
    prefix: Optional[str] = None,
    contains: Optional[str] = None,
    keyword_weights: Optional[Dict[str, float]] = None,
    shuffle: bool = False,
    loop: bool = False,
    seed: Optional[int] = None,
    # Passed through to stream_parquet_rows:
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
      config:          CloudStorageConfig for the bucket.
      keys:            list of parquet keys to stream from.
      keys_generator:  iterator that yields parquet keys.
      prefix:          S3 prefix to list parquet files from.
      contains:        filter listed keys to those containing this substring.
      keyword_weights: dict of {"keyword": weight}. Files matching a keyword
                       are sampled with probability proportional to weight.
                       Files matching no keyword are excluded.
                       e.g. {"yodas": 3.0, "librilight": 1.0, "voxpopuli": 0.5}
      shuffle:         shuffle the key order (for keys/prefix mode).
      loop:            loop forever over the files (for training).
      seed:            random seed for reproducibility.
      batch_size:      pyarrow batch size per row-group read.
      columns:         subset of columns to read.
      bulk:            download whole file at once (vs row-group streaming).
      max_retries:     None = retry forever on network errors.

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
        batch_size=batch_size,
        columns=columns,
        bulk=bulk,
        max_retries=max_retries,
    )

    if keys_generator is not None:
        if loop:
            raise ValueError("loop=True is not supported with keys_generator "
                             "(generator can only be consumed once)")
        for key in keys_generator:
            log.info("Streaming from %s", key)
            yield from stream_parquet_rows(config, key, s3=s3, **stream_kwargs)
        return

    # Apply keyword weights if provided
    if keyword_weights is not None:
        keys, weights = _assign_weights(keys, keyword_weights)
        if not keys:
            raise ValueError("No parquet files matched any keyword in keyword_weights")
        log.info("Weighted sampling: %d files across %d keywords", len(keys), len(keyword_weights))

    rng = random.Random(seed)
    iteration = 0

    while True:
        iteration += 1

        if keyword_weights is not None:
            # Weighted random sampling: pick files proportional to their keyword weight
            order = rng.choices(keys, weights=weights, k=len(keys))
        else:
            order = list(keys)
            if shuffle:
                rng.shuffle(order)

        for key in order:
            log.info("Streaming from %s (iteration %d)", key, iteration)
            yield from stream_parquet_rows(config, key, s3=s3, **stream_kwargs)

        if not loop:
            break


class MultiParquetDataset:
    """PyTorch Dataset that streams from multiple parquet files.

    Ignores index — each call to __getitem__ returns the next row from
    the underlying multi_stream_parquet generator. Set loop=True (default)
    so the generator never exhausts.

    Usage in train_loop config:
        dataset:
          name: train_loop.datasets.multi_parquet_streamer.MultiParquetDataset
          args:
            bucket: granary
            endpoint_url: https://...
            access_key_id: ...
            secret_access_key: ...
            prefix: chunked_parquet/en_yodas/
            keyword_weights:
              yodas: 3.0
              librilight: 1.0
            shuffle: true
            length: 1000000
    """

    def __init__(
        self,
        bucket,
        endpoint_url,
        access_key_id,
        secret_access_key,
        region="auto",
        keys=None,
        prefix=None,
        contains=None,
        keyword_weights=None,
        shuffle=True,
        seed=None,
        loop=True,
        length=1000000,
        batch_size=64,
        columns=None,
        bulk=False,
        max_retries=None,
    ):
        self.length = length
        self._config = CloudStorageConfig(
            bucket=bucket,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
        )
        self._stream_kwargs = dict(
            keys=keys,
            prefix=prefix,
            contains=contains,
            keyword_weights=keyword_weights,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
            batch_size=batch_size,
            columns=columns,
            bulk=bulk,
            max_retries=max_retries,
        )
        self._iter = None

    def _ensure_iter(self):
        if self._iter is None:
            self._iter = iter(multi_stream_parquet(self._config, **self._stream_kwargs))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_iter()
        return next(self._iter)


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
