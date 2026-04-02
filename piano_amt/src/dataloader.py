"""
dataloader.py — DataLoader construction and collation for piano AMT training.

Design:
  - piano_amt_collate: custom collate that stacks tensors and preserves
    audio_path as List[str].
  - get_dataloader: single entry-point that builds Dataset + DataLoader with
    training/validation/test settings.
  - _AugmentedDataset: thin wrapper that applies transforms AFTER __getitem__
    (keeps MAESTRODataset transform-free for caching purposes).
  - sliding_windows: inference utility for segmenting a full spectrogram into
    overlapping/non-overlapping windows.

Papers:
  jongwook/onsets-and-frames src/dataset.py: batch_size=8, num_workers=2.
  Hawthorne 2018a §3: training/validation split semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .constants import MAX_SEGMENT_FRAMES, N_KEYS, N_MELS
from .dataset import MAESTRODataset
from .transforms import Compose, get_train_transform


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def piano_amt_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of dataset items into a batch dict.

    Tensor fields are stacked along a new batch dimension (dim-0).
    The 'audio_path' field is kept as a List[str].

    Args:
        batch: List of dicts, each from MAESTRODataset.__getitem__.

    Returns:
        Dict with:
          mel:        FloatTensor (B, 229, T)
          onset:      FloatTensor (B, T, 88)
          frame:      FloatTensor (B, T, 88)
          offset:     FloatTensor (B, T, 88)
          velocity:   FloatTensor (B, T, 88)
          audio_path: List[str] length B

    Shape:
        mel:      (B, N_MELS, T) = (B, 229, T)
        onset:    (B, T, N_KEYS) = (B, T, 88)
        frame:    (B, T, 88)
        offset:   (B, T, 88)
        velocity: (B, T, 88)
    """
    tensor_keys   = ("mel", "onset", "frame", "offset", "velocity")
    string_keys   = ("audio_path",)

    result: Dict[str, Any] = {}

    for key in tensor_keys:
        tensors = [item[key] for item in batch if key in item]
        if tensors:
            result[key] = torch.stack(tensors, dim=0)

    for key in string_keys:
        values = [item[key] for item in batch if key in item]
        if values:
            result[key] = values

    return result


# ---------------------------------------------------------------------------
# Augmented dataset wrapper
# ---------------------------------------------------------------------------

class _AugmentedDataset(Dataset):
    """
    Thin wrapper that applies a transform pipeline after __getitem__.

    Keeps MAESTRODataset transform-free so its NPZ cache can be reused with
    different augmentation configurations.

    Args:
        dataset:   Base Dataset (MAESTRODataset).
        transform: Callable transform (e.g. Compose from transforms.py).
    """

    def __init__(self, dataset: Dataset, transform: Compose) -> None:
        self.dataset   = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    maestro_root:     Union[str, Path],
    split:            str,
    batch_size:       int = 4,
    num_workers:      int = 2,
    cache_dir:        Optional[Union[str, Path]] = None,
    max_files:        Optional[int] = None,
    use_augmentation: bool = True,
    pin_memory:       bool = True,
    seed:             int = 42,
) -> DataLoader:
    """
    Build a DataLoader for the given MAESTRO split.

    Training split:
      - segment=True (random 640-frame crop per item).
      - shuffle=True, drop_last=True.
      - Augmentation applied if use_augmentation=True.

    Validation/test split:
      - segment=True (random 640-frame crop, same as training).
      - shuffle=False, drop_last=False.
      - No augmentation.

    # Test split:
    #     - segment=False (full spectrogram for sliding-window inference).

    Args:
        maestro_root:     Root directory of MAESTRO dataset.
        split:            "train", "validation", or "test".
        batch_size:       Samples per batch (default 4; jongwook uses 8).
        num_workers:      DataLoader worker processes (default 2).
        cache_dir:        Directory for NPZ cache. Defaults to
                          maestro_root / "cache".
        max_files:        Limit dataset size (for quick testing).
        use_augmentation: If True AND split=="train", wrap with augmentation.
        pin_memory:       Pin tensors to page-locked memory for GPU transfer.
        seed:             Random seed for dataset crops.

    Returns:
        torch.utils.data.DataLoader ready to iterate.

    Papers:
        jongwook/onsets-and-frames: batch_size=8, num_workers=2.
        Hawthorne 2018a §3: train/validation/test split semantics.
    """
    if cache_dir is None:
        cache_dir = Path(maestro_root) / "cache"

    is_train = (split == "train")

    seg_frames = (
        MAX_SEGMENT_FRAMES * 3 if split == "validation"
        else MAX_SEGMENT_FRAMES if is_train
        else None
    )

    dataset = MAESTRODataset(
        maestro_root=maestro_root,
        split=split,
        cache_dir=cache_dir,
        segment=(split != "test"),
        segment_frames=seg_frames,
        max_files=max_files,
        seed=seed,
    )

    # Wrap with augmentation for training
    if is_train and use_augmentation:
        transform = get_train_transform(use_pitch_shift=True)
        dataset   = _AugmentedDataset(dataset, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=piano_amt_collate,
        persistent_workers=(num_workers > 0),
    )

    return loader


# ---------------------------------------------------------------------------
# Sliding-window inference utility
# ---------------------------------------------------------------------------

def sliding_windows(
    mel:           torch.Tensor,
    window_frames: int = MAX_SEGMENT_FRAMES,
    hop_frames:    int = MAX_SEGMENT_FRAMES,
) -> List[Dict[str, Any]]:
    """
    Partition a full-length mel spectrogram into overlapping (or non-overlapping)
    windows for inference.

    The last window is right-padded with zeros if shorter than window_frames.

    Args:
        mel:           FloatTensor (229, T_total) — full log-mel spectrogram.
        window_frames: Width of each window in frames (default 640).
        hop_frames:    Hop between window starts (default 640 = no overlap).

    Returns:
        List of dicts, each containing:
          "mel":         FloatTensor (229, window_frames)
          "start_frame": int — start frame index in the original spectrogram

    Shape:
        Each mel window: (229, window_frames)
    """
    T       = mel.shape[1]
    windows = []
    start   = 0

    while start < T:
        end    = start + window_frames
        chunk  = mel[:, start:end]   # (229, min(window_frames, T-start))

        # Zero-pad last window if shorter than window_frames
        if chunk.shape[1] < window_frames:
            pad_len = window_frames - chunk.shape[1]
            chunk   = torch.nn.functional.pad(chunk, (0, pad_len))

        windows.append({
            "mel":         chunk,          # (229, window_frames)
            "start_frame": start,
        })

        start += hop_frames

    return windows
