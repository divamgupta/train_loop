
import torch
from datasets import load_dataset
from collections import deque
import torch
from .tokenizer import RustBPETokenizer
from train_loop.utils.download import download_file
import pyarrow.parquet as pq
import os
import random

def list_parquet_files(data_dir=None, files_list=None):
    """
    Returns full paths to all parquet files.
    - data_dir: scan a local directory for *.parquet files
    - files_list: list of local paths or URLs; URLs are downloaded to ~/.downloads/
    """
    if files_list is not None:
        paths = []
        for f in files_list:
            local_path = download_file(str(f))
            paths.append(local_path)
        return sorted(paths)
    elif data_dir is not None:
        parquet_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        ])
        parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
        return parquet_paths
    else:
        raise ValueError("Either data_dir or files_list must be provided")

def parquets_iter_batched(data_dir=None, files_list=None, split="train", start=0, step=1, shuffle=True):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - data_dir: local directory of parquet files (mutually exclusive with files_list)
    - files_list: list of local paths or URLs (mutually exclusive with data_dir)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir, files_list=files_list)
    if shuffle:
        random.shuffle(parquet_paths)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        idxs = list(range(start, pf.num_row_groups, step))
        if shuffle:
            random.shuffle(idxs)
        for rg_idx in idxs:
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

def _make_tokenizer(tokenizer_dir=None, tokenizer_name=None):
    """
    Create a tokenizer from either a local directory or a tiktoken pretrained name.
    - tokenizer_dir: local directory with tokenizer.pkl (RustBPETokenizer)
    - tokenizer_name: tiktoken encoding name, e.g. "gpt2" (no files needed)
    """
    if tokenizer_dir is not None:
        return RustBPETokenizer.from_directory(tokenizer_dir)
    elif tokenizer_name is not None:
        return RustBPETokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError("Either tokenizer_dir or tokenizer_name must be provided")

def yield_tokens(parquet_data_dir=None, tokenizer_dir=None, B=None, n_tokens=None, split=None,
                 tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", shuffle=True,
                 parquet_files_list=None, tokenizer_name=None):
    """Stream pretraining text from parquet files, tokenize, yield training batches.

    Data source (one required):
      parquet_data_dir   - local directory of .parquet files
      parquet_files_list - list of local paths or URLs (downloaded automatically)

    Tokenizer (one required):
      tokenizer_dir  - local directory with tokenizer files
      tokenizer_name - tiktoken encoding name, e.g. "gpt2" (no download needed)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    needed_tokens = B * n_tokens + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = _make_tokenizer(tokenizer_dir=tokenizer_dir, tokenizer_name=tokenizer_name)
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(data_dir=parquet_data_dir, files_list=parquet_files_list, split=split, shuffle=shuffle):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for faster transfers between CPU and GPU:
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, n_tokens).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, n_tokens).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets

class BasicTokensDL(torch.utils.data.Dataset):
    def __init__(self, parquet_data_dir=None, parquet_files_list=None,
                 tokenizer_dir=None, tokenizer_name=None,
                 n_tokens=None, split="train", len=1000000):
        """
        Streaming dataset over parquet files for language model pretraining.

        Data source (one required):
          parquet_data_dir   - local directory of .parquet files
          parquet_files_list - list of local paths or URLs (downloaded automatically)

        Tokenizer (one required):
          tokenizer_dir  - local directory with tokenizer files (RustBPETokenizer)
          tokenizer_name - tiktoken encoding name, e.g. "gpt2" (no download needed)
        """
        self.datagen = yield_tokens(
            parquet_data_dir=parquet_data_dir,
            parquet_files_list=parquet_files_list,
            tokenizer_dir=tokenizer_dir,
            tokenizer_name=tokenizer_name,
            B=1,
            n_tokens=n_tokens,
            split=split,
            shuffle=True,
        )
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x , y = next(self.datagen)
        result = {"input_ids": x[0], "gt_output_ids": y[0]}
        return result

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
        gt_output_ids = torch.stack([item["gt_output_ids"] for item in batch], dim=0)
        return {"input_ids": input_ids, "gt_output_ids": gt_output_ids}
