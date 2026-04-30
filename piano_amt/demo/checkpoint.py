"""
demo model loading — checkpoint download, model initialisation, and summary helpers.

Purpose:
  This file handles the model-loading stage of the demo pipeline. It resolves
  the demo checkpoint, downloads it into the local demo checkpoint directory
  when needed, restores the OnsetsAndFrames model weights, moves the model to
  the selected device, and returns the model ready for inference.

Design:
  - resolve_checkpoint_path() ensures demo directories exist and retrieves the
    configured checkpoint file into CHECKPOINT_DIR.
  - load_demo_model() selects CUDA when available, loads the checkpoint, builds
    the model with the configured complexity, restores weights, and switches
    the model to evaluation mode.
  - checkpoint_summary() returns key checkpoint metadata as a dictionary.
  - format_checkpoint_summary() formats the same metadata for notebook or demo
    display.
  - maybe_torchinfo_summary() optionally creates a compact model architecture
    summary when torchinfo is installed.

Outputs:
  - Loaded OnsetsAndFrames model ready for transcription.
  - Checkpoint dictionary for metadata and reproducibility checks.
  - Local checkpoint path used by the demo.
  - Human-readable checkpoint and model summaries for display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from huggingface_hub import hf_hub_download

from demo.demo_config import (
    CHECKPOINT_DIR,
    HF_CHECKPOINT_FILENAME,
    HF_REPO_ID,
    HF_REPO_TYPE,
    MODEL_COMPLEXITY,
    ensure_demo_dirs,
)
from models.onsets_frames.model import OnsetsAndFrames


def resolve_checkpoint_path(force_download: bool = False) -> Path:
    ensure_demo_dirs()
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CHECKPOINT_FILENAME,
        repo_type=HF_REPO_TYPE,
        force_download=force_download,
        local_dir=CHECKPOINT_DIR,
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def load_demo_model(device: str | torch.device | None = None) -> Tuple[OnsetsAndFrames, Dict[str, Any], Path]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_checkpoint_path()
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = OnsetsAndFrames(model_complexity=MODEL_COMPLEXITY)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, ckpt, ckpt_path


def checkpoint_summary(model: OnsetsAndFrames, ckpt: Dict[str, Any], ckpt_path: str | Path) -> Dict[str, Any]:
    return {
        "checkpoint_path": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "global_step": ckpt.get("global_step"),
        "parameter_count": model.count_parameters(),
    }


def format_checkpoint_summary(model: OnsetsAndFrames, ckpt: Dict[str, Any], ckpt_path: str | Path) -> str:
    s = checkpoint_summary(model, ckpt, ckpt_path)
    return (
        f"Checkpoint: {s['checkpoint_path']}\n"
        f"Epoch: {s['epoch']}\n"
        f"Val loss: {s['val_loss']}\n"
        f"Best val loss: {s['best_val_loss']}\n"
        f"Global step: {s['global_step']}\n"
        f"Trainable params: {s['parameter_count']:,}"
    )


def maybe_torchinfo_summary(model: OnsetsAndFrames, n_mels: int = 229, time_steps: int = 512) -> str:
    try:
        from torchinfo import summary
    except Exception:
        return "torchinfo not installed; skipping compact summary."

    try:
        result = summary(
            model,
            input_size=(1, n_mels, time_steps),
            dtypes=[torch.float32],
            verbose=0,
            depth=3,
            col_names=("input_size", "output_size", "num_params"),
        )
        return str(result)
    except Exception as exc:
        return f"torchinfo summary unavailable: {exc}"
