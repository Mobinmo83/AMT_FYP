"""
dataset.py — MAESTRO dataset loading, NPZ caching, and PyTorch Dataset.

Design:
  - NPZ caching: preprocess audio+MIDI once, store on disk.
    Strategy from jongwook/onsets-and-frames src/dataset.py.
  - Random 640-frame crops for training.
    jongwook src/dataset.py: MAX_SEGMENT_FRAMES=640, MAX_SEGMENT_SAMPLES=327680.
  - MAESTRO CSV split column: "train" / "validation" / "test".
    Hawthorne et al. 2018b §3.

Papers:
  Hawthorne 2018a §3: hyperparameters.
  Hawthorne 2018b "MAESTRO": dataset splits.
  jongwook/onsets-and-frames src/dataset.py: NPZ caching + crop strategy.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .audio import load_audio, wav_to_log_mel
from .constants import (
    FRAMES_PER_SECOND,
    MAESTRO_AUDIO_COL,
    MAESTRO_DURATION_COL,
    MAESTRO_MIDI_COL,
    MAESTRO_SPLIT_COL,
    MAX_SEGMENT_FRAMES,
    N_KEYS,
    N_MELS,
    SAMPLE_RATE,
)
from .midi import midi_path_to_rolls


# ---------------------------------------------------------------------------
# Cache path helper
# ---------------------------------------------------------------------------

def _cache_path(audio_path: Union[str, Path], cache_dir: Union[str, Path]) -> Path:
    """
    Compute the NPZ cache file path for a given audio file.

    Args:
        audio_path: Path to the source audio file.
        cache_dir:  Directory where cache files are stored.

    Returns:
        Path: cache_dir / "<audio_stem>.npz"
    """
    stem = Path(audio_path).stem
    return Path(cache_dir) / f"{stem}.npz"


# ---------------------------------------------------------------------------
# Preprocessing and caching
# ---------------------------------------------------------------------------

def preprocess_and_cache(
    audio_path: Union[str, Path],
    midi_path:  Union[str, Path],
    cache_path: Union[str, Path],
) -> None:
    """
    Preprocess one audio+MIDI pair and save the result to an NPZ file.

    Steps:
      1. Load audio (torchaudio) and resample to 16 kHz.
      2. Compute log-mel spectrogram: log(mel + 1e-9).
      3. Derive n_frames from mel shape.
      4. Compute 4-head piano-roll labels from MIDI.
      5. Save all arrays as float32 NPZ.

    NPZ keys:
      mel       — (229, T_frames) log-mel spectrogram
      onset     — (T_frames, 88)  onset piano roll
      frame     — (T_frames, 88)  frame piano roll
      offset    — (T_frames, 88)  offset piano roll
      velocity  — (T_frames, 88)  velocity piano roll
      sr        — scalar int, sample rate (always 16000)

    Args:
        audio_path: Path to audio file.
        midi_path:  Path to MIDI file.
        cache_path: Destination .npz file path.

    Papers:
        jongwook/onsets-and-frames src/dataset.py: NPZ caching strategy.
        Hawthorne 2018a §3.1: label encoding.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load audio
    waveform, sr = load_audio(audio_path, target_sr=SAMPLE_RATE, mono=True)

    # 2. Compute log-mel spectrogram on CPU
    device = torch.device("cpu")
    log_mel = wav_to_log_mel(waveform, device=device)  # (229, T)

    n_frames = log_mel.shape[1]

    # 3. Compute piano-roll labels
    onset, frame, offset, velocity = midi_path_to_rolls(
        midi_path, n_frames=n_frames, start_sec=0.0, duration_sec=None
    )

    # 4. Save to NPZ (float32 for all arrays)
    np.savez_compressed(
        str(cache_path),
        mel=log_mel.numpy().astype(np.float32),        # (229, T)
        onset=onset.numpy().astype(np.float32),         # (T, 88)
        frame=frame.numpy().astype(np.float32),         # (T, 88)
        offset=offset.numpy().astype(np.float32),       # (T, 88)
        velocity=velocity.numpy().astype(np.float32),   # (T, 88)
        sr=np.array(sr, dtype=np.int32),
    )


def load_from_cache(cache_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a preprocessed NPZ cache file into a dict of Tensors.

    Args:
        cache_path: Path to .npz file produced by preprocess_and_cache().

    Returns:
        Dict with keys: mel, onset, frame, offset, velocity, sr.
        All values are float32 Tensors (sr is a scalar int Tensor).

    Shape:
        mel:      (229, T_frames)
        onset:    (T_frames, 88)
        frame:    (T_frames, 88)
        offset:   (T_frames, 88)
        velocity: (T_frames, 88)
    """
    data = np.load(str(cache_path))
    return {
        "mel":      torch.from_numpy(data["mel"]).float(),
        "onset":    torch.from_numpy(data["onset"]).float(),
        "frame":    torch.from_numpy(data["frame"]).float(),
        "offset":   torch.from_numpy(data["offset"]).float(),
        "velocity": torch.from_numpy(data["velocity"]).float(),
        "sr":       torch.tensor(int(data["sr"]), dtype=torch.int32),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MAESTRODataset(Dataset):
    """
    PyTorch Dataset for the MAESTRO v3 dataset.

    Reads the maestro-v3.0.0.csv split manifest, resolves absolute paths,
    and lazily loads/caches preprocessed NPZ files on first access.

    Each item returned by __getitem__ contains:
      mel:        FloatTensor (229, 640)  — log-mel spectrogram segment
      onset:      FloatTensor (640, 88)  — onset piano roll
      frame:      FloatTensor (640, 88)  — frame piano roll
      offset:     FloatTensor (640, 88)  — offset piano roll
      velocity:   FloatTensor (640, 88)  — velocity piano roll
      audio_path: str                    — source audio file path

    Args:
        maestro_root: Root directory of the MAESTRO dataset (contains CSV).
        split:        One of "train", "validation", "test".
        cache_dir:    Directory for NPZ cache files.
        segment:      If True, return random 640-frame crops (training mode).
                      If False, return the full spectrogram (evaluation mode).
        max_files:    If set, limit dataset to this many files (for quick tests).
        seed:         Random seed for reproducible crops.

    Papers:
        Hawthorne 2018b "MAESTRO" §3: dataset splits and CSV format.
        jongwook/onsets-and-frames: NPZ caching + 640-frame crop strategy.
    """

    def __init__(
        self,
        maestro_root:   Union[str, Path],
        split:          str,
        cache_dir:      Union[str, Path],
        segment:        bool = True,
        segment_frames: Optional[int] = None,
        max_files:      Optional[int] = None,
        seed:           int = 42,
    ) -> None:
        super().__init__()

        self.maestro_root = Path(maestro_root)
        self.split        = split
        self.cache_dir    = Path(cache_dir)
        self.segment      = segment
        self.seed         = seed
        self.segment_frames = segment_frames or MAX_SEGMENT_FRAMES

        # Locate CSV (MAESTRO v3 ships a single CSV at the root)
        csv_files = sorted(self.maestro_root.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV found in {self.maestro_root}. "
                "Make sure MAESTRO v3 is downloaded and extracted."
            )
        df = pd.read_csv(csv_files[0])

        # Filter by split
        if MAESTRO_SPLIT_COL not in df.columns:
            raise KeyError(
                f"Column '{MAESTRO_SPLIT_COL}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        df = df[df[MAESTRO_SPLIT_COL] == split].reset_index(drop=True)

        if max_files is not None:
            df = df.iloc[:max_files]

        # Build list of (audio_path, midi_path) records
        self.records: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            audio_path = str(self.maestro_root / row[MAESTRO_AUDIO_COL])
            midi_path  = str(self.maestro_root / row[MAESTRO_MIDI_COL])
            self.records.append({
                "audio_path": audio_path,
                "midi_path":  midi_path,
            })

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Seed RNG for reproducible crops
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Load (and cache if needed) one item.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict with keys: mel, onset, frame, offset, velocity, audio_path.

        Shape:
            mel:      (229, 640) if segment=True else (229, T)
            onset:    (640, 88)  if segment=True else (T, 88)
            frame:    (640, 88)  if segment=True else (T, 88)
            offset:   (640, 88)  if segment=True else (T, 88)
            velocity: (640, 88)  if segment=True else (T, 88)
        """
        record = self.records[idx]
        audio_path = record["audio_path"]
        midi_path  = record["midi_path"]

        cp = _cache_path(audio_path, self.cache_dir)

        if not cp.exists():
            preprocess_and_cache(audio_path, midi_path, cp)

        data = load_from_cache(cp)

        if self.segment:
            data = self._random_segment(data)

        data["audio_path"] = audio_path
        return data

    def _random_segment(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract a random MAX_SEGMENT_FRAMES-length crop from data tensors.

        If the piece is shorter than MAX_SEGMENT_FRAMES, the tensors are
        zero-padded on the right to reach exactly MAX_SEGMENT_FRAMES.

        Args:
            data: Dict with mel (229, T) and roll tensors (T, 88).

        Returns:
            Dict with the same keys, all cropped/padded to MAX_SEGMENT_FRAMES.

        Papers:
            jongwook/onsets-and-frames src/dataset.py: 640-frame random crop.
        """
        mel = data["mel"]         # (229, T)
        T   = mel.shape[1]
        W   = self.segment_frames

        if T <= W:
            # Zero-pad to W frames
            pad_cols = W - T
            mel_out  = torch.nn.functional.pad(mel, (0, pad_cols))  # (229, W)
            result   = {"mel": mel_out}
            for key in ("onset", "frame", "offset", "velocity"):
                tensor = data[key]  # (T, 88)
                padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad_cols))  # (W, 88)
                result[key] = padded
        else:
            # Random crop
            start = self._rng.randint(0, T - W)
            mel_out = mel[:, start : start + W]   # (229, W)
            result  = {"mel": mel_out}
            for key in ("onset", "frame", "offset", "velocity"):
                result[key] = data[key][start : start + W, :]  # (W, 88)

        return result


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(
    maestro_root: Union[str, Path],
    cache_dir:    Union[str, Path],
    splits:       Sequence[str] = ("train", "validation", "test"),
) -> None:
    """
    Preprocess and cache all files in the given MAESTRO splits.

    Skips files whose NPZ cache already exists. Prints per-file errors
    without stopping the full run.

    Args:
        maestro_root: Root directory of the MAESTRO dataset.
        cache_dir:    Directory to store NPZ cache files.
        splits:       Tuple of split names to process.

    Papers:
        jongwook/onsets-and-frames src/dataset.py: NPZ caching strategy.
    """
    maestro_root = Path(maestro_root)
    cache_dir    = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(maestro_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {maestro_root}.")
    df = pd.read_csv(csv_files[0])

    for split in splits:
        split_df = df[df[MAESTRO_SPLIT_COL] == split].reset_index(drop=True)
        n = len(split_df)
        print(f"[{split}]: {n} files")

        errors = 0
        for _, row in tqdm(split_df.iterrows(), total=n, desc=f"Cache {split}"):
            audio_path = str(maestro_root / row[MAESTRO_AUDIO_COL])
            midi_path  = str(maestro_root / row[MAESTRO_MIDI_COL])
            cp = _cache_path(audio_path, cache_dir)

            if cp.exists():
                continue

            try:
                preprocess_and_cache(audio_path, midi_path, cp)
            except Exception as exc:
                errors += 1
                print(f"\n  ERROR processing {audio_path}: {exc}")

        if errors:
            print(f"  {errors} error(s) in split '{split}'")
        else:
            print(f"  Split '{split}' cached successfully.")
