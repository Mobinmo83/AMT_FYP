"""
transforms.py — Data augmentation transforms for piano AMT training.

Design:
  - All transforms operate on Dict[str, Tensor] batches (same format as
    MAESTRODataset.__getitem__ returns).
  - RandomPitchShift: shifts both mel bins AND label columns simultaneously
    to preserve alignment.  From KinWaiCheuk/ICPR2020 augmentation strategy.
  - SpecAugment-style masking: time and frequency masking on mel only (or
    jointly on mel + labels for time masking).
    Paper: KinWaiCheuk/ICPR2020 github for generalisation augmentation.

Papers:
  KinWaiCheuk/ICPR2020: RandomPitchShift ±1 semitone, SpecAugment masking.
  Hawthorne 2018a §3: BINS_PER_SEMITONE derived from N_MELS/N_KEYS.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from .constants import N_KEYS, N_MELS

# Number of mel bins per semitone.
# N_MELS=229 filterbank spans 88 semitones (A0–C8 ≈ piano range).
# KinWaiCheuk/ICPR2020: BINS_PER_SEMITONE = N_MELS / N_KEYS ≈ 2.60
BINS_PER_SEMITONE: float = N_MELS / N_KEYS  # ≈ 2.6022

# Keys used in a data dict that carry piano-roll labels (dim-1 = key axis)
LABEL_KEYS = ("onset", "frame", "offset", "velocity")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Transform(ABC):
    """Abstract base class for all AMT data transforms."""

    @abstractmethod
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply the transform to a data dictionary.

        Args:
            data: Dict with keys: mel (229, T), onset (T, 88), frame (T, 88),
                  offset (T, 88), velocity (T, 88).  May also contain 'audio_path'.

        Returns:
            Transformed data dict (same keys).
        """


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose(Transform):
    """
    Apply a sequence of transforms in order.

    Args:
        transforms: List of Transform objects to chain.
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


# ---------------------------------------------------------------------------
# RandomPitchShift
# ---------------------------------------------------------------------------

class RandomPitchShift(Transform):
    """
    Randomly shift pitch by ±max_shift semitones.

    Shifts the mel spectrogram along the frequency axis (dim-0) AND shifts
    all label tensors along the key axis (dim-1) by the same amount, so
    audio-label alignment is perfectly preserved.

    Out-of-range bins/columns are zeroed (no wrap-around).

    Source: KinWaiCheuk/ICPR2020 GitHub — pitch augmentation strategy.
    Hawthorne 2018a §3: mel bin ↔ semitone relationship.

    Args:
        max_shift: Maximum pitch shift in semitones (default 1).
                   A random integer in [-max_shift, +max_shift] is sampled.
        p:         Probability of applying the transform (default 0.5).
    """

    def __init__(self, max_shift: int = 1, p: float = 0.5) -> None:
        self.max_shift = max_shift
        self.p         = p

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() >= self.p:
            return data

        shift_keys = random.randint(-self.max_shift, self.max_shift)
        if shift_keys == 0:
            return data

        data = {k: v for k, v in data.items()}  # shallow copy

        # --- Shift mel along frequency (dim-0) ---
        mel = data["mel"].clone()   # (229, T)
        mel_bins = int(round(shift_keys * BINS_PER_SEMITONE))

        if mel_bins > 0:
            # Shift up: move bins to higher indices, zero-pad bottom
            mel[mel_bins:, :] = mel[:-mel_bins, :].clone()
            mel[:mel_bins, :] = 0.0
        elif mel_bins < 0:
            # Shift down: move bins to lower indices, zero-pad top
            shift_abs = -mel_bins
            mel[:-shift_abs, :] = mel[shift_abs:, :].clone()
            mel[-shift_abs:, :] = 0.0

        data["mel"] = mel

        # --- Shift label tensors along key axis (dim-1) ---
        for key in LABEL_KEYS:
            if key not in data:
                continue
            tensor = data[key].clone()  # (T, 88)

            if shift_keys > 0:
                # Shift right: higher pitches — zero left columns
                tensor[:, shift_keys:] = tensor[:, :-shift_keys].clone()
                tensor[:, :shift_keys] = 0.0
            else:
                # Shift left: lower pitches — zero right columns
                shift_abs = -shift_keys
                tensor[:, :-shift_abs] = tensor[:, shift_abs:].clone()
                tensor[:, -shift_abs:] = 0.0

            data[key] = tensor

        return data

    def __repr__(self) -> str:
        return f"RandomPitchShift(max_shift={self.max_shift}, p={self.p})"


# ---------------------------------------------------------------------------
# RandomTimeMask
# ---------------------------------------------------------------------------

class RandomTimeMask(Transform):
    """
    Zero out a contiguous block of frames in the mel AND all label tensors.

    SpecAugment-style time masking applied jointly to mel and labels so that
    the model does not learn from artificially masked regions.

    Source: KinWaiCheuk/ICPR2020 GitHub — SpecAugment time masking.

    Args:
        max_mask_frames: Maximum width of the masked time block (default 50).
        p:               Probability of applying the transform (default 0.3).
    """

    def __init__(self, max_mask_frames: int = 50, p: float = 0.3) -> None:
        self.max_mask_frames = max_mask_frames
        self.p               = p

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() >= self.p:
            return data

        mel = data["mel"]     # (229, T)
        T   = mel.shape[1]

        mask_len   = random.randint(1, min(self.max_mask_frames, T))
        mask_start = random.randint(0, T - mask_len)

        data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        # Zero mel time block
        data["mel"][:, mask_start : mask_start + mask_len] = 0.0

        # Zero matching label blocks
        for key in LABEL_KEYS:
            if key in data:
                data[key][mask_start : mask_start + mask_len, :] = 0.0

        return data

    def __repr__(self) -> str:
        return (
            f"RandomTimeMask(max_mask_frames={self.max_mask_frames}, p={self.p})"
        )


# ---------------------------------------------------------------------------
# RandomFreqMask
# ---------------------------------------------------------------------------

class RandomFreqMask(Transform):
    """
    Zero out a contiguous frequency band in the mel spectrogram only.

    SpecAugment-style frequency masking.  Does NOT modify labels — frequency
    masking does not change which pitches are active.

    Source: KinWaiCheuk/ICPR2020 GitHub — SpecAugment frequency masking.

    Args:
        max_mask_bins: Maximum height of the masked frequency band (default 20).
        p:             Probability of applying the transform (default 0.3).
    """

    def __init__(self, max_mask_bins: int = 20, p: float = 0.3) -> None:
        self.max_mask_bins = max_mask_bins
        self.p             = p

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() >= self.p:
            return data

        mel  = data["mel"]    # (229, T)
        F    = mel.shape[0]   # = 229

        mask_len   = random.randint(1, min(self.max_mask_bins, F))
        mask_start = random.randint(0, F - mask_len)

        data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
        data["mel"][mask_start : mask_start + mask_len, :] = 0.0

        return data

    def __repr__(self) -> str:
        return f"RandomFreqMask(max_mask_bins={self.max_mask_bins}, p={self.p})"


# ---------------------------------------------------------------------------
# RandomGainJitter
# ---------------------------------------------------------------------------

class RandomGainJitter(Transform):
    """
    Apply random additive gain in log space to the mel spectrogram.

    Because the mel is already log-compressed, adding a constant in log space
    is equivalent to multiplying the mel magnitudes by a random scalar.
    This simulates recording-level variation.

    Args:
        gain_range: (min_dB, max_dB) range for additive log gain (default ±3 dB).
        p:          Probability of applying the transform (default 0.3).
    """

    def __init__(
        self,
        gain_range: tuple[float, float] = (-3.0, 3.0),
        p: float = 0.3,
    ) -> None:
        self.gain_range = gain_range
        self.p          = p

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() >= self.p:
            return data

        gain = random.uniform(self.gain_range[0], self.gain_range[1])
        data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
        data["mel"] = data["mel"] + gain
        return data

    def __repr__(self) -> str:
        return f"RandomGainJitter(gain_range={self.gain_range}, p={self.p})"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_train_transform(use_pitch_shift: bool = True) -> Compose:
    """
    Return the standard training augmentation pipeline.

    Augmentations (in order):
      1. RandomPitchShift(max_shift=1, p=0.5)  — KinWaiCheuk/ICPR2020
      2. RandomTimeMask(max_mask_frames=50, p=0.3)  — KinWaiCheuk/ICPR2020
      3. RandomFreqMask(max_mask_bins=20, p=0.3)    — KinWaiCheuk/ICPR2020
      4. RandomGainJitter(gain_range=(-3.0, 3.0), p=0.3)

    Args:
        use_pitch_shift: If False, omit RandomPitchShift (useful for ablations).

    Returns:
        Compose transform ready to apply to training batches.
    """
    transforms: List[Transform] = []

    if use_pitch_shift:
        transforms.append(RandomPitchShift(max_shift=1, p=0.5))

    transforms += [
        RandomTimeMask(max_mask_frames=50, p=0.3),
        RandomFreqMask(max_mask_bins=20, p=0.3),
        RandomGainJitter(gain_range=(-3.0, 3.0), p=0.3),
    ]

    return Compose(transforms)


def get_val_transform() -> None:
    """
    Return the validation transform (None = no augmentation).

    Validation uses the raw preprocessed data — no augmentation is applied.

    Returns:
        None
    """
    return None
