from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from demo.demo_config import MANIFEST_PATH, SAMPLE_RATE, TEMP_DIR, UPLOADED_DIR, ensure_demo_dirs
from src.audio import load_audio_as_log_mel


def load_sample_manifest(manifest_path: str | Path | None = None) -> Dict:
    path = Path(manifest_path or MANIFEST_PATH)
    if not path.exists():
        return {"samples": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_demo_sample_names(manifest_path: str | Path | None = None) -> List[str]:
    manifest = load_sample_manifest(manifest_path)
    return [item["name"] for item in manifest.get("samples", [])]


def resolve_demo_sample_paths(sample_name: str, repo_root: str | Path | None = None) -> Tuple[Path, Path]:
    manifest = load_sample_manifest()
    base = Path(repo_root) if repo_root is not None else Path(".")
    for item in manifest.get("samples", []):
        if item["name"] == sample_name:
            audio_path = (base / item["audio"]).resolve()
            labels_path = (base / item["labels"]).resolve()
            return audio_path, labels_path
    raise KeyError(f"Sample not found in manifest: {sample_name}")


def save_uploaded_audio(upload_bytes: bytes, filename: str) -> Path:
    """Save raw upload bytes and normalise to 16 kHz mono WAV."""
    ensure_demo_dirs()
    raw_path = UPLOADED_DIR / Path(filename).name
    with open(raw_path, "wb") as f:
        f.write(upload_bytes)

    normalised_path = UPLOADED_DIR / f"{raw_path.stem}_16k_mono.wav"
    y, _ = librosa.load(str(raw_path), sr=SAMPLE_RATE, mono=True)
    sf.write(str(normalised_path), y, SAMPLE_RATE)
    return normalised_path


def audio_path_to_mel(audio_path: str | Path) -> torch.Tensor:
    """Load audio and return mel with shape (229, T)."""
    mel = load_audio_as_log_mel(str(audio_path))
    if isinstance(mel, dict):
        mel = mel["mel"]
    elif isinstance(mel, tuple):
        mel = mel[0]
    mel = torch.tensor(np.asarray(mel), dtype=torch.float32)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel array, got {tuple(mel.shape)}")
    if mel.shape[0] != 229 and mel.shape[1] == 229:
        mel = mel.transpose(0, 1)
    return mel


def load_ground_truth_labels(labels_path: str | Path) -> Dict[str, torch.Tensor]:
    data = np.load(str(labels_path))
    return {
        "onset": torch.tensor(data["onset"], dtype=torch.float32),
        "frame": torch.tensor(data["frame"], dtype=torch.float32),
        "offset": torch.tensor(data["offset"], dtype=torch.float32),
        "velocity": torch.tensor(data["velocity"], dtype=torch.float32),
    }
