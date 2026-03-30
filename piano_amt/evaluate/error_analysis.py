"""
evaluate/error_analysis.py — Targeted error analysis beyond standard F1.

Computes metrics specifically relevant for:
  - Understanding offset timing errors (offset jitter)
  - Chord completeness (important for the reconstruction phase)
  - Duplicate onset detection (common failure mode)

Three metrics:
  1. duplicate_note_rate:
       Fraction of predicted onsets that occur within 50 ms of another
       predicted onset on the same pitch. High value → model hallucinates
       repeated notes.

  2. chord_completeness:
       For each ground-truth simultaneous group (chord), what fraction of
       chord members were correctly detected?  Mean over all chords in file.
       A chord is defined as ≥2 notes with onsets within 50 ms of each other.

  3. offset_mae_ms:
       Mean absolute error (milliseconds) between predicted and ground-truth
       note offsets, for notes where the onset was correctly detected.
       This measures offset timing accuracy.

  4. onset_mae_ms:
       Mean absolute error (milliseconds) for correctly matched onsets.
       Baseline reference.

Papers:
  Hawthorne 2018a §4: note evaluation context.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Tolerance for "same onset" (50 ms standard)
ONSET_TOLERANCE_SEC = 0.05
MIN_CHORD_SIZE      = 2          # minimum simultaneous notes to count as chord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roll_to_events(
    onset_roll:  torch.Tensor,  # (T, 88)
    offset_roll: torch.Tensor,  # (T, 88) or None
    fps:         float,
    threshold:   float = 0.5,
) -> List[Tuple[float, float, int]]:
    """
    Convert onset (and optional offset) rolls to event list.

    Returns:
        List of (onset_sec, offset_sec, pitch_midi) sorted by onset.
        If offset_roll is None, offset_sec = onset_sec (onset-only mode).
    """
    from src.constants import MIN_MIDI, N_KEYS
    from models.onsets_frames.decode import rolls_to_note_events

    if offset_roll is None:
        offset_roll = torch.zeros_like(onset_roll)

    events = rolls_to_note_events(
        onset_roll=onset_roll,
        frame_roll=onset_roll,   # use onset as frame proxy for onset-only mode
        velocity_roll=torch.zeros_like(onset_roll),
        fps=fps,
        onset_threshold=threshold,
        frame_threshold=threshold,
    )
    return [(e.onset_sec, e.offset_sec, e.pitch) for e in events]


# ---------------------------------------------------------------------------
# Metric 1: Duplicate note rate
# ---------------------------------------------------------------------------

def compute_duplicate_note_rate(
    pred_onset: torch.Tensor,  # (T, 88)
    fps:        float,
    threshold:  float = 0.5,
) -> float:
    """
    Fraction of predicted onsets that are duplicates of another onset
    on the same pitch within ONSET_TOLERANCE_SEC.

    Args:
        pred_onset: Predicted onset roll (T, 88), values in [0,1].
        fps:        Frames per second.
        threshold:  Onset detection threshold.

    Returns:
        Duplicate rate in [0, 1].  0 = no duplicates, 1 = all duplicates.
    """
    from src.constants import N_KEYS, MIN_MIDI

    onset_np  = (pred_onset > threshold).cpu().numpy()
    T, K      = onset_np.shape
    tol_frames = int(ONSET_TOLERANCE_SEC * fps)

    n_total = 0
    n_dup   = 0

    for key in range(K):
        onset_frames = np.where(onset_np[:, key])[0]
        n_total += len(onset_frames)
        for i, f in enumerate(onset_frames):
            # Check if any previous onset on same key is within tolerance
            if i > 0 and (f - onset_frames[i - 1]) <= tol_frames:
                n_dup += 1

    return float(n_dup / n_total) if n_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 2: Chord completeness
# ---------------------------------------------------------------------------

def compute_chord_completeness(
    pred_onset: torch.Tensor,  # (T, 88)
    gt_onset:   torch.Tensor,  # (T, 88)
    fps:        float,
    threshold:  float = 0.5,
) -> float:
    """
    Mean fraction of chord members detected, across all ground-truth chords.

    Algorithm:
      1. Group GT onsets that occur within ONSET_TOLERANCE_SEC into chords.
         A "chord" is any group of ≥ MIN_CHORD_SIZE notes.
      2. For each chord, count how many of its members have a matching
         predicted onset within ONSET_TOLERANCE_SEC on the same pitch.
      3. Return mean completeness across all chords.

    Args:
        pred_onset: Predicted onset roll (T, 88).
        gt_onset:   Ground-truth onset roll (T, 88).
        fps:        Frames per second.
        threshold:  Onset detection threshold.

    Returns:
        Mean chord completeness in [0, 1].
    """
    from src.constants import N_KEYS, MIN_MIDI

    tol_frames = int(ONSET_TOLERANCE_SEC * fps)

    gt_np   = (gt_onset   > 0.5).cpu().numpy()
    pred_np = (pred_onset  > threshold).cpu().numpy()
    T, K    = gt_np.shape

    # Collect gt onset events as (frame, key) pairs
    gt_events = sorted(
        [(f, k) for f in range(T) for k in range(K) if gt_np[f, k]],
        key=lambda e: e[0]
    )

    if len(gt_events) < MIN_CHORD_SIZE:
        return 1.0  # no chords to evaluate

    # Group events by time into chords
    chords: List[List[Tuple[int, int]]] = []
    current_chord: List[Tuple[int, int]] = [gt_events[0]]

    for f, k in gt_events[1:]:
        if f - current_chord[0][0] <= tol_frames:
            current_chord.append((f, k))
        else:
            if len(current_chord) >= MIN_CHORD_SIZE:
                chords.append(current_chord)
            current_chord = [(f, k)]

    if len(current_chord) >= MIN_CHORD_SIZE:
        chords.append(current_chord)

    if not chords:
        return 1.0

    # For each chord, compute fraction of members detected
    completeness_scores: List[float] = []
    for chord in chords:
        detected = 0
        for (gt_f, gt_k) in chord:
            # Check if any predicted onset on same key within tolerance
            f_lo = max(0, gt_f - tol_frames)
            f_hi = min(T,  gt_f + tol_frames + 1)
            if pred_np[f_lo:f_hi, gt_k].any():
                detected += 1
        completeness_scores.append(detected / len(chord))

    return float(np.mean(completeness_scores))


# ---------------------------------------------------------------------------
# Metric 3: Offset MAE
# ---------------------------------------------------------------------------

def compute_offset_mae(
    pred_onset:  torch.Tensor,  # (T, 88)
    pred_frame:  torch.Tensor,  # (T, 88)
    pred_offset: torch.Tensor,  # (T, 88)
    gt_onset:    torch.Tensor,  # (T, 88)
    gt_frame:    torch.Tensor,  # (T, 88)
    gt_offset:   torch.Tensor,  # (T, 88)
    fps:         float,
    onset_threshold:  float = 0.5,
    offset_threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute mean absolute onset and offset errors (in milliseconds) for
    correctly matched notes.

    A predicted note matches a ground-truth note when:
      - Same pitch
      - Onset within ONSET_TOLERANCE_SEC

    Args:
        pred_onset, pred_offset: Predicted rolls (T, 88).
        gt_onset, gt_offset:     Ground-truth rolls (T, 88).
        fps:                     Frames per second.
        onset_threshold:         Threshold for onset detection.
        offset_threshold:        Threshold for offset detection.

    Returns:
        (onset_mae_ms, offset_mae_ms)  — floats in milliseconds.
    """
    from src.constants import N_KEYS, MIN_MIDI, VELOCITY_SCALE
    from models.onsets_frames.decode import rolls_to_note_events

    pred_events = rolls_to_note_events(
        onset_roll=pred_onset,
        frame_roll=pred_frame,
        velocity_roll=torch.zeros_like(pred_onset),
        fps=fps,
        onset_threshold=onset_threshold,
        frame_threshold=onset_threshold,
    )
    gt_events = rolls_to_note_events(
        onset_roll=gt_onset,
        frame_roll=gt_frame,
        velocity_roll=torch.zeros_like(gt_onset),
        fps=fps,
        onset_threshold=0.5,
        frame_threshold=0.5,
    )

    # Build GT dict keyed by pitch for fast lookup
    gt_by_pitch: Dict[int, List] = {}
    for e in gt_events:
        gt_by_pitch.setdefault(e.pitch, []).append(e)

    onset_errors:  List[float] = []
    offset_errors: List[float] = []

    for pred_e in pred_events:
        candidates = gt_by_pitch.get(pred_e.pitch, [])
        for gt_e in candidates:
            if abs(pred_e.onset_sec - gt_e.onset_sec) <= ONSET_TOLERANCE_SEC:
                onset_errors.append(
                    abs(pred_e.onset_sec - gt_e.onset_sec) * 1000.0
                )
                offset_errors.append(
                    abs(pred_e.offset_sec - gt_e.offset_sec) * 1000.0
                )
                break  # match first within tolerance

    onset_mae  = float(np.mean(onset_errors))  if onset_errors  else 0.0
    offset_mae = float(np.mean(offset_errors)) if offset_errors else 0.0
    return onset_mae, offset_mae


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_error_analysis(
    pred_onset:       torch.Tensor,
    pred_frame:       torch.Tensor,
    pred_offset:      torch.Tensor,
    gt_onset:         torch.Tensor,
    gt_frame:         torch.Tensor,
    gt_offset:        torch.Tensor,
    fps:              float,
    onset_threshold:  float = 0.5,
) -> Dict[str, float]:
    """
    Run all three error-analysis metrics for one file.

    Returns:
        Dict with keys:
          duplicate_note_rate   — [0,1]
          chord_completeness    — [0,1]
          onset_mae_ms          — float
          offset_mae_ms         — float  ← offset timing accuracy
    """
    dup_rate = compute_duplicate_note_rate(
        pred_onset=pred_onset, fps=fps, threshold=onset_threshold
    )
    chord_comp = compute_chord_completeness(
        pred_onset=pred_onset, gt_onset=gt_onset,
        fps=fps, threshold=onset_threshold
    )
    onset_mae, offset_mae = compute_offset_mae(
        pred_onset=pred_onset,
        pred_frame=pred_frame,
        pred_offset=pred_offset,
        gt_onset=gt_onset,
        gt_frame=gt_frame,
        gt_offset=gt_offset,
        fps=fps,
        onset_threshold=onset_threshold,
    )

    return {
        "duplicate_note_rate": dup_rate,
        "chord_completeness":  chord_comp,
        "onset_mae_ms":        onset_mae,
        "offset_mae_ms":       offset_mae,
    }
