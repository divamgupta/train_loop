#!/usr/bin/env python3
"""
Generic single-thread generator to stream rows from a parquet object in any
S3-compatible store (R2, S3, ...), without the huggingface `datasets` dependency:

  * row-group streaming via HTTP range reads (never downloads the whole file),
  * every network read retried (by default FOREVER, with capped backoff) on
    hang / disconnect / 5xx / rate-limit — built for long training runs.

    from stream_parquet import stream_parquet_rows, CloudStorageConfig
    cfg = CloudStorageConfig(bucket="granary", endpoint_url="https://...",
                             access_key_id="...", secret_access_key="...")
    for row in stream_parquet_rows(cfg, "chunked_parquet/.../..._chunk_000400.parquet"):
        text = row["text"]

Everything (fetch, parquet decode) runs in the calling thread.
Audio decoding is NOT handled here — use MultiParquetAudioDataset for that.
"""
import io
import logging
import time
from dataclasses import dataclass

import boto3
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import (
    ClientError, ConnectionError as BotoConnectionError,
    HTTPClientError, IncompleteReadError,
)

log = logging.getLogger("stream_parquet")

# Match on the base classes, not the leaves: a mid-body disconnect surfaces as
# ResponseStreamingError (botocore wraps urllib3's ProtocolError), which is an
# HTTPClientError and was NOT covered by listing the leaf classes individually.
#   BotoConnectionError -> EndpointConnectionError, ConnectTimeoutError, ...
#   HTTPClientError     -> ReadTimeoutError, ConnectionClosedError,
#                          ResponseStreamingError, ...
#   IncompleteReadError  = stream ended cleanly but short of Content-Length
# The builtin ConnectionError is a subclass of OSError, so it needs no entry.
_RETRYABLE = (BotoConnectionError, HTTPClientError, IncompleteReadError,
              TimeoutError, OSError)
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

    def __init__(self, s3, bucket, key, block=8 << 20, max_retries=None,
                 max_range=32 << 20, chunk=4 << 20):
        """
        block:     read-ahead size for small reads (one cached block).
        max_range: largest span fetched per HTTP request. A parquet file written
                   as a single row group makes pyarrow ask for the whole audio
                   column at once (~260 MB); splitting that into windows keeps
                   any one dropped connection cheap.
        chunk:     incremental body read size, i.e. resume granularity.
        """
        self.s3, self.bucket, self.key = s3, bucket, key
        self.block, self.max_retries = block, max_retries
        self.max_range, self.chunk = max_range, chunk
        self.pos = 0
        self.size = _with_retries(
            lambda: s3.head_object(Bucket=bucket, Key=key)["ContentLength"],
            max_retries, what="head_object")
        self._cache = b""
        self._cache_start = 0
        self._closed = False

    def _get_window(self, start, end_inclusive):
        """Fetch one window, resuming in place across retries.

        The body is drained incrementally into `buf`, so a mid-transfer
        disconnect keeps what already arrived and the retry re-issues the GET
        from the first byte still missing instead of restarting the window.
        """
        buf = bytearray()
        expected = end_inclusive - start + 1

        def fetch():
            if len(buf) >= expected:
                return          # already complete; a retry here would be an invalid range
            if buf:
                log.warning("stream_parquet: resuming bytes=%d-%d at +%d/%d bytes",
                            start, end_inclusive, len(buf), expected)
            rng = f"bytes={start + len(buf)}-{end_inclusive}"
            body = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=rng)["Body"]
            try:
                while True:
                    part = body.read(self.chunk)
                    if not part:
                        break
                    buf.extend(part)
            finally:
                body.close()

        _with_retries(fetch, self.max_retries,
                      what=f"read bytes={start}-{end_inclusive}")

        if len(buf) != expected:
            raise IOError(f"{self.key}: short read for bytes={start}-{end_inclusive} "
                          f"({len(buf)} of {expected} bytes)")
        return bytes(buf)

    def _get_range(self, start, end_inclusive):
        """Fetch [start, end_inclusive] as bytes, one bounded window at a time."""
        if end_inclusive - start + 1 <= self.max_range:
            return self._get_window(start, end_inclusive)
        out = bytearray()
        pos = start
        while pos <= end_inclusive:
            win_end = min(pos + self.max_range - 1, end_inclusive)
            out.extend(self._get_window(pos, win_end))
            pos = win_end + 1
        return bytes(out)

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
        if want >= self.block:
            # Large read (e.g. a whole column chunk): hand it straight back rather
            # than also parking a second copy of it in the cache.
            data = self._get_range(self.pos, end - 1)
            self.pos += len(data)
            return data
        fetch_end = min(self.pos + self.block, self.size) - 1
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


# ── the generator ─────────────────────────────────────────────────────────────

def stream_parquet_rows(config: CloudStorageConfig, key, s3=None,
                        batch_size=64, columns=None,
                        bulk=False, max_retries=None):
    """
    Stream rows from `key` (full object path) in `config.bucket`. Yields one dict per
    row with raw column values (no audio decoding — handle that downstream).

    config:  CloudStorageConfig (bucket + endpoint + credentials).
    key:     full object key, e.g. "chunked_parquet/<folder>/<name>.parquet".
    s3:      pass a make_client(config) to reuse it across many files.
    bulk:    True = one retried GET into memory (fastest for ~200 MB files);
             False (default) = row-group streaming via ranged reads.
    columns: subset to fetch, e.g. ['audio','text'].
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
    row_idx = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            cols = batch.to_pydict()
            keys = list(cols.keys())
            for i in range(batch.num_rows):
                row = {k: cols[k][i] for k in keys}
                row["__source_parquet__"] = key
                row["__row_index__"] = row_idx
                row_idx += 1
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