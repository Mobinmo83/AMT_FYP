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
    """Download the public checkpoint from Hugging Face into the local cache.

    Returns a local file path that can be passed directly to ``torch.load``.
    """
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
    """Load your public checkpoint into the existing OnsetsAndFrames model."""
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
    """Return a compact torchinfo summary string when torchinfo is installed."""
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
