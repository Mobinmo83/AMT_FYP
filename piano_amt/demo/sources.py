"""
demo sample loading — manifest access, uploaded audio handling, and mel preparation.

Purpose:
  This file loads the audio inputs used by the public demo. It supports both
  prepared demo examples from the sample manifest and custom user-uploaded
  audio files. It also converts selected audio into the log-mel tensor required
  by the model and loads cached ground-truth label rolls when available.

Design:
  - load_sample_manifest() reads sample_manifest.json and returns an empty
    sample list if the manifest is not present.
  - list_demo_sample_names() exposes the available prepared examples for
    notebook dropdowns or demo controls.
  - resolve_demo_sample_paths() returns the audio path, label-roll path,
    optional original MIDI path, and metadata for a selected demo sample.
  - save_uploaded_audio() stores an uploaded file, converts it to 16 kHz mono
    WAV, and returns the normalised audio path.
  - audio_path_to_mel() converts any selected audio file into a contiguous
    (229, T) log-mel tensor for full-length inference.
  - load_ground_truth_labels() loads cached onset, frame, offset, and velocity
    rolls for quantitative comparison.

Outputs:
  - Resolved demo sample paths and metadata.
  - Normalised uploaded audio files in the demo upload directory.
  - Model-ready log-mel spectrogram tensors.
  - Ground-truth label dictionaries for evaluation and visual comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from demo.demo_config import MANIFEST_PATH, SAMPLE_RATE, UPLOADED_DIR, ensure_demo_dirs
from src.audio import load_audio_as_log_mel


def load_sample_manifest(manifest_path: str | Path | None = None) -> Dict:
    path = Path(manifest_path or MANIFEST_PATH)
    if not path.exists():
        return {"samples": []}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("samples", [])
    return manifest


def list_demo_sample_names(manifest_path: str | Path | None = None) -> List[str]:
    manifest = load_sample_manifest(manifest_path)
    return [item["name"] for item in manifest.get("samples", [])]


def get_demo_sample_manifest_item(sample_name: str, manifest_path: str | Path | None = None) -> Dict:
    manifest = load_sample_manifest(manifest_path)
    for item in manifest.get("samples", []):
        if item.get("name") == sample_name:
            return item
    raise KeyError(f"Sample not found in manifest: {sample_name}")


def resolve_demo_sample_paths(sample_name: str, repo_root: str | Path | None = None) -> Tuple[Path, Path, Path | None, Dict]:
    """Return audio, label-roll, optional original-MIDI path, and metadata."""
    item = get_demo_sample_manifest_item(sample_name)
    base = Path(repo_root) if repo_root is not None else Path(".")
    audio_path = (base / item["audio"]).resolve()
    labels_path = (base / item["labels"]).resolve()
    midi_rel = item.get("midi") or item.get("original_midi")
    midi_path = (base / midi_rel).resolve() if midi_rel else None
    metadata = dict(item.get("metadata", {}))
    metadata.update({k: v for k, v in item.items() if k not in {"audio", "labels", "midi", "original_midi", "metadata"}})
    return audio_path, labels_path, midi_path, metadata


def save_uploaded_audio(upload_bytes: bytes, filename: str) -> Path:
    ensure_demo_dirs()
    raw_path = UPLOADED_DIR / Path(filename).name
    with open(raw_path, "wb") as f:
        f.write(upload_bytes)
    normalised_path = UPLOADED_DIR / f"{raw_path.stem}_16k_mono.wav"
    y, _ = librosa.load(str(raw_path), sr=SAMPLE_RATE, mono=True)
    sf.write(str(normalised_path), y, SAMPLE_RATE)
    return normalised_path


def audio_path_to_mel(audio_path: str | Path) -> torch.Tensor:
    mel = load_audio_as_log_mel(str(audio_path))
    if isinstance(mel, dict):
        mel = mel.get("mel", next(iter(mel.values())))
    elif isinstance(mel, tuple):
        mel = mel[0]
    mel = torch.tensor(np.asarray(mel), dtype=torch.float32)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel array, got {tuple(mel.shape)}")
    if mel.shape[0] != 229 and mel.shape[1] == 229:
        mel = mel.transpose(0, 1)
    if mel.shape[0] != 229:
        raise ValueError(f"Expected mel shape (229, T), got {tuple(mel.shape)}")
    return mel.contiguous()


def load_ground_truth_labels(labels_path: str | Path) -> Dict[str, torch.Tensor]:
    data = np.load(str(labels_path))
    required = ["onset", "frame", "offset", "velocity"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Ground-truth label file is missing keys: {missing}")
    return {k: torch.tensor(data[k], dtype=torch.float32) for k in required}
