#!/usr/bin/env python3
"""
Generic single-thread generator to stream rows (with decoded audio) from a parquet
object in any S3-compatible store (R2, S3, ...), without the huggingface `datasets`
or `torchcodec` dependencies — but matching their streaming robustness:

  * row-group streaming via HTTP range reads (never downloads the whole file),
  * every network read retried (by default FOREVER, with capped backoff) on
    hang / disconnect / 5xx / rate-limit — built for long training runs,
  * lazy per-row audio decode with soundfile (robust for Opus via libsndfile).

    from stream_parquet import stream_parquet_rows, CloudStorageConfig
    cfg = CloudStorageConfig(bucket="granary", endpoint_url="https://...",
                             access_key_id="...", secret_access_key="...")
    for row in stream_parquet_rows(cfg, "chunked_parquet/.../..._chunk_000400.parquet"):
        wav = row["audio"]["array"]          # float32 np.ndarray, mono
        sr  = row["audio"]["sampling_rate"]
        text = row["text"]

Everything (fetch, parquet decode, audio decode) runs in the calling thread.
"""
import io
import logging
import time
from dataclasses import dataclass

import boto3
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from botocore.config import Config
from botocore.exceptions import (
    ClientError, ConnectionError as BotoConnectionError,
    ReadTimeoutError, ConnectTimeoutError, ConnectionClosedError,
    IncompleteReadError, EndpointConnectionError,
)

log = logging.getLogger("stream_parquet")

_RETRYABLE = (BotoConnectionError, EndpointConnectionError, ReadTimeoutError,
              ConnectTimeoutError, ConnectionClosedError, IncompleteReadError,
              ConnectionError, TimeoutError, OSError)
_RETRY_CODES = {"500", "502", "503", "504", "SlowDown", "ServiceUnavailable",
                "RequestTimeout", "RequestTimeoutException", "InternalError",
                "ThrottlingException", "Throttling"}


# ── storage config + client ───────────────────────────────────────────────────

@dataclass
class CloudStorageConfig:
    """Connection details for an S3-compatible bucket (R2, S3, MinIO, ...).

    """
    bucket: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region: str = "auto"
    read_timeout: int = 60         # bounds a silent socket hang -> retryable timeout
    connect_timeout: int = 15
    max_pool_connections: int = 8


def make_client(config: CloudStorageConfig):
    """Build a boto3 S3 client from a CloudStorageConfig. Reuse it across files."""
    return boto3.client(
        "s3", endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name=config.region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      max_pool_connections=config.max_pool_connections,
                      read_timeout=config.read_timeout,
                      connect_timeout=config.connect_timeout),
    )


# ── retry ─────────────────────────────────────────────────────────────────────

def _is_retryable(e):
    if isinstance(e, _RETRYABLE):
        return True
    if isinstance(e, ClientError):
        return e.response.get("Error", {}).get("Code") in _RETRY_CODES
    return False


def _with_retries(fn, max_retries=None, base=0.5, max_delay=30.0, what="request"):
    """Retry a network op on hang / disconnect / 5xx / rate-limit with capped
    exponential backoff.

    max_retries=None  -> retry FOREVER on network errors (never give up), for long
                         training runs. A genuinely non-network error (404, 403,
                         parse error, ...) is raised immediately so real bugs aren't
                         masked. Every retry is logged so stalls are visible.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e):
                raise                                   # real error -> surface it
            if max_retries is not None and attempt >= max_retries:
                raise
            code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else None
            cap = max(max_delay, 60.0) if code in {"SlowDown", "503", "ServiceUnavailable",
                                                    "ThrottlingException"} else max_delay
            delay = min(base * 2 ** min(attempt - 1, 20), cap)
            log.warning("stream_parquet: %s failed (%s: %s); retry %d in %.1fs",
                        what, type(e).__name__, str(e)[:120], attempt, delay)
            time.sleep(delay)


# ── seekable, retrying, block-cached file over an S3 object ───────────────────

class RobustS3File:
    """Seekable, retrying, block-cached file over an S3/R2 object — feeds pyarrow.
    Reads are served from a one-block read-ahead cache; misses do a ranged GET with
    retries so a mid-stream disconnect resumes from the failed offset."""

    def __init__(self, s3, bucket, key, block=8 << 20, max_retries=None):
        self.s3, self.bucket, self.key = s3, bucket, key
        self.block, self.max_retries = block, max_retries
        self.pos = 0
        self.size = _with_retries(
            lambda: s3.head_object(Bucket=bucket, Key=key)["ContentLength"],
            max_retries, what="head_object")
        self._cache = b""
        self._cache_start = 0
        self._closed = False

    def _get_range(self, start, end_inclusive):
        rng = f"bytes={start}-{end_inclusive}"
        return _with_retries(
            lambda: self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=rng)["Body"].read(),
            self.max_retries, what=f"read {rng}")

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        end = self.size if (n is None or n < 0) else min(self.pos + n, self.size)
        want = end - self.pos
        if self._cache and self._cache_start <= self.pos and end <= self._cache_start + len(self._cache):
            off = self.pos - self._cache_start
            data = self._cache[off:off + want]
            self.pos += len(data)
            return data
        fetch_end = min(max(end, self.pos + self.block), self.size) - 1
        self._cache = self._get_range(self.pos, fetch_end)
        self._cache_start = self.pos
        data = self._cache[:want]
        self.pos += len(data)
        return data

    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else (self.pos + pos if whence == 1 else self.size + pos)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    @property
    def closed(self):
        return self._closed

    def close(self):
        self._closed = True
        self._cache = b""


# ── audio decode ──────────────────────────────────────────────────────────────

def _decode_audio(a, target_sr=None):
    if not a or a.get("bytes") is None:
        return {"array": np.zeros(0, np.float32), "sampling_rate": target_sr or 16000, "path": None}
    arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    if target_sr and target_sr != sr:
        import soxr
        arr = soxr.resample(arr, sr, target_sr).astype(np.float32)
        sr = target_sr
    return {"array": arr, "sampling_rate": int(sr), "path": a.get("path")}


# ── the generator ─────────────────────────────────────────────────────────────

def stream_parquet_rows(config: CloudStorageConfig, key, s3=None, decode_audio=True,
                        target_sampling_rate=None, batch_size=64, columns=None,
                        bulk=False, max_retries=None):
    """
    Stream rows from `key` (full object path) in `config.bucket`. Yields one dict per
    row; 'audio' -> {'array': float32 np.ndarray (mono), 'sampling_rate': int,
    'path': str} when decode_audio, else raw {'bytes','path'}.

    config:  CloudStorageConfig (bucket + endpoint + credentials).
    key:     full object key, e.g. "chunked_parquet/<folder>/<name>.parquet".
    s3:      pass a make_client(config) to reuse it across many files.
    bulk:    True = one retried GET into memory (fastest for ~200 MB files);
             False (default) = row-group streaming via ranged reads.
    columns: subset to fetch, e.g. ['audio','text'].
    target_sampling_rate: resample audio to this rate (requires `soxr`).
    max_retries: None (default) = retry network errors FOREVER (never give up), for
                 long training runs. Set an int to cap. Genuine errors always raise.
    """
    s3 = s3 or make_client(config)
    bucket = config.bucket

    if bulk:
        body = _with_retries(lambda: s3.get_object(Bucket=bucket, Key=key)["Body"].read(),
                             max_retries, what="get_object")
        src = io.BytesIO(body)
    else:
        src = RobustS3File(s3, bucket, key, max_retries=max_retries)

    pf = pq.ParquetFile(src)
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            cols = batch.to_pydict()
            keys = list(cols.keys())
            for i in range(batch.num_rows):
                row = {k: cols[k][i] for k in keys}
                if decode_audio and "audio" in row:
                    row["audio"] = _decode_audio(row["audio"], target_sampling_rate)
                yield row
    finally:
        if isinstance(src, RobustS3File):
            src.close()


if __name__ == "__main__":
    import os, sys

    cfg = CloudStorageConfig(
        bucket=os.environ["CLOUD_STORAGE_BUCKET"],
        endpoint_url=os.environ["CLOUD_STORAGE_ENDPOINT_URL"],
        access_key_id=os.environ["CLOUD_STORAGE_ACCESS_KEY_ID"],
        secret_access_key=os.environ["CLOUD_STORAGE_SECRET_ACCESS_KEY"],
    )

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not arg:
        print("Usage: python -m train_loop.datasets.stream_parquet <parquet_key>")
        sys.exit(1)

    n = 0
    t0 = time.time()
    prev = t0
    for row in stream_parquet_rows(cfg, arg):
        n += 1
        now = time.time()
        a = row.get("audio", {})
        sr = a.get("sampling_rate", 0)
        dur = len(a.get("array", [])) / sr if sr else 0
        print(f"[{n}] +{(now - prev) * 1000:6.1f}ms  {sr}Hz "
              f"{dur:.2f}s | {row.get('text', '')[:60]!r}")
        prev = now
    dt = time.time() - t0
    print(f"streamed {n} rows from {arg} in {dt:.1f}s ({n / dt:.1f} rows/s)")