"""
verify_pipeline.py — End-to-end pipeline verification script.

Runs 5 ordered checks that assert exact tensor shapes and sanity conditions.
Intended to be run at the start of every Colab session to confirm the full
pipeline is intact from constants through DataLoader.

Usage:
    python scripts/verify_pipeline.py --maestro_root /path/to/maestro-v3.0.0

All checks print "[N] name" then "  ✓ OK" or raise AssertionError.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure piano_amt/src is importable when run from any directory
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Check 1: Constants
# ---------------------------------------------------------------------------

def check_constants() -> None:
    """
    Assert that all critical constants have the exact values mandated by
    Hawthorne 2018a §3 and jongwook/onsets-and-frames.
    """
    print("[1] check_constants")
    from src.constants import (
        FRAMES_PER_SECOND,
        HOP_LENGTH,
        MAX_SEGMENT_FRAMES,
        MIN_MIDI,
        MAX_MIDI,
        N_KEYS,
        N_MELS,
        SAMPLE_RATE,
    )

    assert N_MELS == 229, f"N_MELS={N_MELS}, expected 229"
    assert SAMPLE_RATE == 16000, f"SAMPLE_RATE={SAMPLE_RATE}, expected 16000"
    assert HOP_LENGTH == 512, f"HOP_LENGTH={HOP_LENGTH}, expected 512"
    assert abs(FRAMES_PER_SECOND - 31.25) < 1e-6, (
        f"FRAMES_PER_SECOND={FRAMES_PER_SECOND}, expected 31.25"
    )
    assert MAX_SEGMENT_FRAMES == 640, (
        f"MAX_SEGMENT_FRAMES={MAX_SEGMENT_FRAMES}, expected 640"
    )
    assert N_KEYS == 88, f"N_KEYS={N_KEYS}, expected 88"
    assert MIN_MIDI == 21, f"MIN_MIDI={MIN_MIDI}, expected 21"
    assert MAX_MIDI == 108, f"MAX_MIDI={MAX_MIDI}, expected 108"

    print("  ✓ OK")


# ---------------------------------------------------------------------------
# Check 2: Audio loading and mel computation
# ---------------------------------------------------------------------------

def check_audio(audio_path: str) -> None:
    """
    Verify audio loading and log-mel spectrogram extraction.

    Asserts:
      - Waveform shape is (1, N) for some N > 0.
      - Sample rate is 16000 after resampling.
      - Log-mel shape[0] == 229 (N_MELS).
      - audio_samples_to_frames round-trips correctly.
    """
    print("[2] check_audio")
    from src.audio import (
        audio_samples_to_frames,
        frames_to_audio_samples,
        load_audio,
        wav_to_log_mel,
    )
    from src.constants import N_MELS, SAMPLE_RATE, HOP_LENGTH

    # Load
    waveform, sr = load_audio(audio_path, target_sr=SAMPLE_RATE, mono=True)
    assert waveform.dim() == 2, f"waveform.dim()={waveform.dim()}, expected 2"
    assert waveform.shape[0] == 1, (
        f"waveform.shape[0]={waveform.shape[0]}, expected 1 (mono)"
    )
    assert waveform.shape[1] > 0, "waveform has zero samples"
    assert sr == SAMPLE_RATE, f"sr={sr}, expected {SAMPLE_RATE}"

    # Mel
    log_mel = wav_to_log_mel(waveform, device=torch.device("cpu"))
    assert log_mel.dim() == 2, f"log_mel.dim()={log_mel.dim()}, expected 2"
    assert log_mel.shape[0] == N_MELS, (
        f"log_mel.shape[0]={log_mel.shape[0]}, expected {N_MELS}"
    )

    # Frame conversion round-trip
    n_samples = waveform.shape[1]
    n_frames  = audio_samples_to_frames(n_samples)
    assert n_frames > 0, "audio_samples_to_frames returned 0"
    reconstructed = frames_to_audio_samples(n_frames)
    assert abs(reconstructed - n_samples) <= HOP_LENGTH, (
        f"frames_to_audio_samples({n_frames})={reconstructed} "
        f"differs from {n_samples} by more than HOP_LENGTH={HOP_LENGTH}"
    )

    print("  ✓ OK")


# ---------------------------------------------------------------------------
# Check 3: MIDI loading and roll encoding
# ---------------------------------------------------------------------------

def check_midi(midi_path: str, n_frames: int) -> None:
    """
    Verify MIDI loading and 4-head piano-roll encoding.

    Asserts:
      - All 4 roll tensors have shape (n_frames, 88).
      - onset_roll has at least one active entry (piece is not silent).
    """
    print("[3] check_midi")
    from src.constants import N_KEYS
    from src.midi import midi_path_to_rolls

    onset, frame, offset, velocity = midi_path_to_rolls(
        midi_path, n_frames=n_frames, start_sec=0.0, duration_sec=None
    )

    expected_shape = (n_frames, N_KEYS)
    for name, tensor in [("onset", onset), ("frame", frame),
                         ("offset", offset), ("velocity", velocity)]:
        assert tensor.shape == expected_shape, (
            f"{name}.shape={tuple(tensor.shape)}, expected {expected_shape}"
        )
        assert tensor.dtype == torch.float32, (
            f"{name}.dtype={tensor.dtype}, expected float32"
        )

    assert onset.sum() > 0, (
        "onset_roll is all zeros — no notes detected. Check MIDI file."
    )

    print("  ✓ OK")


# ---------------------------------------------------------------------------
# Check 4: MAESTRODataset __getitem__
# ---------------------------------------------------------------------------

def check_dataset(maestro_root: str, max_files: int = 3) -> None:
    """
    Verify the MAESTRODataset returns correctly shaped items.

    Asserts:
      - All required keys are present.
      - mel.shape == (229, 640).
      - All roll shapes == (640, 88).
    """
    print("[4] check_dataset")
    from src.constants import MAX_SEGMENT_FRAMES, N_KEYS, N_MELS
    from src.dataset import MAESTRODataset

    cache_dir = Path(maestro_root) / "_verify_cache"

    ds = MAESTRODataset(
        maestro_root=maestro_root,
        split="train",
        cache_dir=str(cache_dir),
        segment=True,
        max_files=max_files,
    )
    assert len(ds) > 0, "Dataset is empty for split='train'"

    item = ds[0]

    required_keys = ("mel", "onset", "frame", "offset", "velocity", "audio_path")
    for key in required_keys:
        assert key in item, f"Key '{key}' missing from dataset item"

    assert item["mel"].shape == (N_MELS, MAX_SEGMENT_FRAMES), (
        f"mel.shape={tuple(item['mel'].shape)}, "
        f"expected ({N_MELS}, {MAX_SEGMENT_FRAMES})"
    )
    for key in ("onset", "frame", "offset", "velocity"):
        expected = (MAX_SEGMENT_FRAMES, N_KEYS)
        assert item[key].shape == expected, (
            f"{key}.shape={tuple(item[key].shape)}, expected {expected}"
        )
    assert isinstance(item["audio_path"], str), "audio_path should be a str"

    print("  ✓ OK")


# ---------------------------------------------------------------------------
# Check 5: DataLoader batch shapes
# ---------------------------------------------------------------------------

def check_dataloader(maestro_root: str, max_files: int = 3) -> None:
    """
    Verify that get_dataloader produces batches with the correct shapes.

    Asserts:
      - mel batch shape == (B, 229, 640).
      - All roll batch shapes == (B, 640, 88).
    """
    print("[5] check_dataloader")
    from src.constants import MAX_SEGMENT_FRAMES, N_KEYS, N_MELS
    from src.dataloader import get_dataloader

    cache_dir = Path(maestro_root) / "_verify_cache"

    loader = get_dataloader(
        maestro_root=maestro_root,
        split="train",
        batch_size=2,
        num_workers=0,
        cache_dir=str(cache_dir),
        max_files=max_files,
        use_augmentation=False,
        pin_memory=False,
    )
    assert len(loader) > 0, "DataLoader has zero batches"

    batch = next(iter(loader))
    B = batch["mel"].shape[0]

    assert batch["mel"].shape == (B, N_MELS, MAX_SEGMENT_FRAMES), (
        f"mel.shape={tuple(batch['mel'].shape)}, "
        f"expected ({B}, {N_MELS}, {MAX_SEGMENT_FRAMES})"
    )
    for key in ("onset", "frame", "offset", "velocity"):
        expected = (B, MAX_SEGMENT_FRAMES, N_KEYS)
        assert batch[key].shape == expected, (
            f"{key}.shape={tuple(batch[key].shape)}, expected {expected}"
        )
    assert "audio_path" in batch, "audio_path missing from batch"
    assert len(batch["audio_path"]) == B, (
        f"len(audio_path)={len(batch['audio_path'])}, expected {B}"
    )

    print("  ✓ OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run all 5 pipeline verification checks in order.

    Reads the first available train-split audio+MIDI pair from the MAESTRO
    CSV to use as inputs for checks 2 and 3.
    """
    parser = argparse.ArgumentParser(
        description="Verify the piano AMT preprocessing pipeline."
    )
    parser.add_argument(
        "--maestro_root", required=True, type=str,
        help="Root directory of the MAESTRO v3 dataset (contains *.csv)"
    )
    parser.add_argument(
        "--max_files", type=int, default=3,
        help="Max files to load for dataset/dataloader checks (default 3)"
    )
    args = parser.parse_args()

    maestro_root = Path(args.maestro_root)

    # Find CSV and extract first train row for checks 2 + 3
    csv_files = sorted(maestro_root.glob("*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: No CSV found in {maestro_root}")

    from src.constants import MAESTRO_AUDIO_COL, MAESTRO_MIDI_COL, MAESTRO_SPLIT_COL

    df = pd.read_csv(csv_files[0])
    train_rows = df[df[MAESTRO_SPLIT_COL] == "train"].reset_index(drop=True)
    if train_rows.empty:
        sys.exit("ERROR: No train rows found in CSV.")

    first_row  = train_rows.iloc[0]
    audio_path = str(maestro_root / first_row[MAESTRO_AUDIO_COL])
    midi_path  = str(maestro_root / first_row[MAESTRO_MIDI_COL])

    print("=" * 60)
    print("Piano AMT Pipeline Verification")
    print(f"MAESTRO root: {maestro_root}")
    print(f"Test file  : {Path(audio_path).name}")
    print("=" * 60)

    # ------- Run checks -------
    check_constants()

    check_audio(audio_path)

    # For check_midi we need n_frames — derive from mel shape
    from src.audio import load_audio, wav_to_log_mel
    from src.constants import SAMPLE_RATE
    waveform, _ = load_audio(audio_path, target_sr=SAMPLE_RATE, mono=True)
    log_mel = wav_to_log_mel(waveform, device=torch.device("cpu"))
    n_frames = log_mel.shape[1]

    check_midi(midi_path, n_frames=n_frames)

    check_dataset(str(maestro_root), max_files=args.max_files)

    check_dataloader(str(maestro_root), max_files=args.max_files)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
