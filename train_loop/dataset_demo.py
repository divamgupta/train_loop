#!/usr/bin/env python
"""
train_loop.dataset_demo — interactive Gradio browser for any train_loop dataset config.

Usage
-----
    python -m train_loop.dataset_demo configs/my_config.yml
    python -m train_loop.dataset_demo configs/my_config.yml --port 7861 --share

The YAML must contain a top-level ``dataset:`` key (same format used by train_loop).
"""

import argparse
import json
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from train_loop.utils.dynamic_import import build_class


# ─────────────────────────────────────────────────────────────────────────────
# Type detection
# ─────────────────────────────────────────────────────────────────────────────

def _looks_like_audio(key: str, val) -> bool:
    low = key.lower()
    if any(w in low for w in ("audio", "wav", "waveform", "speech")):
        return isinstance(val, (torch.Tensor, np.ndarray))
    if not isinstance(val, (torch.Tensor, np.ndarray)):
        return False
    arr = val.numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    if arr.ndim == 1 and arr.shape[0] > 1000:
        return True
    if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > 1000:
        return True
    return False


def _looks_like_image(key: str, val) -> bool:
    low = key.lower()
    if any(w in low for w in ("image", "img", "photo", "frame", "mel", "spec")):
        return isinstance(val, (torch.Tensor, np.ndarray))
    if not isinstance(val, (torch.Tensor, np.ndarray)):
        return False
    arr = val.numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return True
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return True
    if arr.ndim == 2 and arr.shape[0] > 8:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Value converters
# ─────────────────────────────────────────────────────────────────────────────

def _to_audio_numpy(val, sample_rate: int):
    """Return (sample_rate, float32_mono_array) for gr.Audio."""
    arr = val.detach().float().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    arr = np.clip(arr.flatten(), -1.0, 1.0)
    return sample_rate, arr


def _to_image_numpy(val):
    """Return uint8 HWC numpy array for gr.Image."""
    arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8) if hi > lo else np.zeros_like(arr, dtype=np.uint8)
    return arr


def _to_text(val) -> str:
    if isinstance(val, (torch.Tensor, np.ndarray)):
        arr = val.numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        return f"Tensor  shape={list(arr.shape)}  dtype={arr.dtype}  min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}"
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, indent=2, default=str)[:4000]
        except Exception:
            return str(val)[:4000]
    return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import gradio as gr

    parser = argparse.ArgumentParser(description="Gradio dataset browser")
    parser.add_argument("config", help="Path to YAML config with a dataset: key")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    assert "dataset" in cfg, "Config must have a top-level 'dataset:' key"

    print(f"Loading dataset from {args.config} …")
    dataset = build_class(OmegaConf.to_container(cfg.dataset, resolve=True))
    n = len(dataset)
    print(f"Dataset length: {n}")

    sample = dataset[0]
    all_keys = list(sample.keys())
    auto_audio = [k for k, v in sample.items() if _looks_like_audio(k, v)]
    auto_image = [k for k, v in sample.items() if _looks_like_image(k, v)]

    with gr.Blocks(title="Dataset Demo") as demo:
        gr.Markdown(
            f"## Dataset Demo\n"
            f"**Config:** `{args.config}`  •  **Length:** {n}  •  "
            f"**Keys:** {', '.join(f'`{k}`' for k in all_keys) or '(none)'}"
        )

        with gr.Row():
            idx_slider = gr.Slider(minimum=0, maximum=max(n - 1, 0), step=1,
                                   value=0, label="Index")
            load_btn = gr.Button("Load", variant="primary", scale=0)

        with gr.Row():
            audio_keys_box = gr.Textbox(
                label="Audio keys (comma-separated)",
                placeholder="e.g. audio_24000, waveform",
                value=", ".join(auto_audio),
                scale=2,
            )
            image_keys_box = gr.Textbox(
                label="Image keys (comma-separated)",
                placeholder="e.g. image, mel_spec",
                value=", ".join(auto_image),
                scale=2,
            )
            sr_box = gr.Number(label="Sample rate", value=24000, scale=1)

        state = gr.State({"idx": 0,
                          "audio_keys": set(auto_audio),
                          "image_keys": set(auto_image),
                          "sr": 24000})

        def commit(idx, ak_str, ik_str, sr):
            return {"idx": int(idx),
                    "audio_keys": {k.strip() for k in ak_str.split(",") if k.strip()},
                    "image_keys": {k.strip() for k in ik_str.split(",") if k.strip()},
                    "sr": int(sr),
                    "_n": random.random()}

        ctrl_inputs = [idx_slider, audio_keys_box, image_keys_box, sr_box]
        load_btn.click(fn=commit, inputs=ctrl_inputs, outputs=state)

        @gr.render(inputs=state)
        def show(s):
            try:
                item = dataset[s["idx"]]
            except Exception as e:
                gr.Markdown(f"**Error loading index {s['idx']}:** {e}")
                return

            for key, val in item.items():
                is_audio = key in s["audio_keys"]
                is_image = key in s["image_keys"]

                if is_audio and isinstance(val, (torch.Tensor, np.ndarray)):
                    try:
                        gr.Audio(value=_to_audio_numpy(val, s["sr"]), label=key, interactive=False)
                    except Exception as e:
                        gr.Textbox(value=f"Audio error: {e}", label=key, interactive=False)

                elif is_image and isinstance(val, (torch.Tensor, np.ndarray)):
                    try:
                        gr.Image(value=_to_image_numpy(val), label=key, interactive=False)
                    except Exception as e:
                        gr.Textbox(value=f"Image error: {e}", label=key, interactive=False)

                else:
                    gr.Textbox(value=_to_text(val), label=key, lines=1, max_lines=10, interactive=False)

        demo.load(fn=lambda ak, ik, sr: commit(0, ak, ik, sr),
                  inputs=[audio_keys_box, image_keys_box, sr_box],
                  outputs=state)

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
