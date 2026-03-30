"""
evaluate/metrics.py — AMT evaluation metrics using mir_eval.

Computes the standard set of metrics used in Hawthorne 2018a and all
subsequent AMT papers for fair comparison:

  Frame-level (binary piano roll):
    frame_precision, frame_recall, frame_f1, frame_accuracy

  Note-level (matched by onset time):
    onset_precision, onset_recall, onset_f1

  Note-level (matched by onset + offset):
    note_with_offset_precision, note_with_offset_recall, note_with_offset_f1

  Note-level (matched by onset + offset + velocity):
    note_with_offset_vel_precision, note_with_offset_vel_recall,
    note_with_offset_vel_f1

All note-level metrics use mir_eval.transcription which matches notes with
an onset tolerance of 50 ms and offset tolerance of 50 ms or 20% of note
duration (whichever is larger) — standard AMT benchmark tolerances.

Dependencies:
    pip install mir_eval

Papers:
    Hawthorne 2018a §4 — evaluation protocol.
    mir_eval: Raffel et al. 2014, "mir_eval: A Transparent Implementation of
              Common MIR Metrics"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Optional mir_eval — raises a clear error if missing
try:
    import mir_eval
    from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
    _MIR_EVAL_AVAILABLE = True
except ImportError:
    _MIR_EVAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper: rolls → mir_eval intervals + pitches
# ---------------------------------------------------------------------------

def _rolls_to_intervals_pitches_velocities(
    onset_roll:      torch.Tensor,   # (T, 88)
    frame_roll:      torch.Tensor,   # (T, 88)
    velocity_roll:   torch.Tensor,   # (T, 88)
    fps:             float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert piano rolls to mir_eval format arrays.

    Returns:
        intervals:  (N, 2) float array [[onset_sec, offset_sec], ...]
        pitches:    (N,)   float array of MIDI note numbers (float for mir_eval)
        velocities: (N,)   float array of velocities [0..127]
    """
    from src.constants import MIN_MIDI, VELOCITY_SCALE
    from models.onsets_frames.decode import rolls_to_note_events

    events = rolls_to_note_events(
        onset_roll=onset_roll,
        frame_roll=frame_roll,
        velocity_roll=velocity_roll,
        fps=fps,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    if not events:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    intervals   = np.array([[e.onset_sec, e.offset_sec] for e in events])
    pitches     = np.array([float(e.pitch) for e in events])
    velocities  = np.array([float(e.velocity) for e in events])
    return intervals, pitches, velocities


# ---------------------------------------------------------------------------
# Frame-level metrics
# ---------------------------------------------------------------------------

def compute_frame_metrics(
    pred_frame:     torch.Tensor,   # (T, 88)
    gt_frame:       torch.Tensor,   # (T, 88)
    frame_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Binary piano-roll frame-level precision / recall / F1 / accuracy.

    Uses the same definition as Hawthorne 2018a §4:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2PR / (P+R)
        A = TP / (TP + FP + FN)   ← note: NOT standard accuracy

    Args:
        pred_frame:      Model output (T, 88), values in [0,1].
        gt_frame:        Ground truth (T, 88), binary.
        frame_threshold: Binarisation threshold for pred_frame.

    Returns:
        Dict with frame_precision, frame_recall, frame_f1, frame_accuracy.
    """
    pred_bin = (pred_frame > frame_threshold).float()
    gt_bin   = (gt_frame   > 0.5).float()

    tp = (pred_bin * gt_bin).sum().item()
    fp = (pred_bin * (1 - gt_bin)).sum().item()
    fn = ((1 - pred_bin) * gt_bin).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "frame_precision": precision,
        "frame_recall":    recall,
        "frame_f1":        f1,
        "frame_accuracy":  accuracy,
    }


# ---------------------------------------------------------------------------
# Note-level metrics (requires mir_eval)
# ---------------------------------------------------------------------------

def compute_note_metrics(
    pred_onset:       torch.Tensor,  # (T, 88)
    pred_frame:       torch.Tensor,
    pred_offset:      torch.Tensor,
    pred_velocity:    torch.Tensor,
    gt_onset:         torch.Tensor,
    gt_frame:         torch.Tensor,
    gt_offset:        torch.Tensor,
    gt_velocity:      torch.Tensor,
    onset_threshold:  float = 0.5,
    frame_threshold:  float = 0.5,
    offset_threshold: float = 0.5,
    fps:              float = 31.25,
) -> Dict[str, float]:
    """
    Note-level precision / recall / F1 using mir_eval.transcription.

    Three tiers:
      1. onset_f1:               onset within 50 ms
      2. note_with_offset_f1:    onset + offset within tolerance
      3. note_with_offset_vel_f1: onset + offset + velocity within 0.1

    Returns dict with all three tiers' P/R/F1 values.
    """
    if not _MIR_EVAL_AVAILABLE:
        raise ImportError(
            "mir_eval is required for note-level metrics. "
            "Install with: pip install mir_eval"
        )

    # Predicted notes
    pred_intervals, pred_pitches, pred_vels = _rolls_to_intervals_pitches_velocities(
        onset_roll=pred_onset,
        frame_roll=pred_frame,
        velocity_roll=pred_velocity,
        fps=fps,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    # Ground-truth notes
    gt_intervals, gt_pitches, gt_vels = _rolls_to_intervals_pitches_velocities(
        onset_roll=gt_onset,
        frame_roll=gt_frame,
        velocity_roll=gt_velocity,
        fps=fps,
        onset_threshold=0.5,
        frame_threshold=0.5,
    )

    results: Dict[str, float] = {}

    if len(pred_pitches) == 0 or len(gt_pitches) == 0:
        for k in [
            "onset_precision","onset_recall","onset_f1",
            "note_with_offset_precision","note_with_offset_recall","note_with_offset_f1",
            "note_with_offset_vel_precision","note_with_offset_vel_recall","note_with_offset_vel_f1",
        ]:
            results[k] = 0.0
        return results

    # Tier 1: onset only (50 ms tolerance)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=gt_intervals,
        ref_pitches=gt_pitches,
        est_intervals=pred_intervals,
        est_pitches=pred_pitches,
        onset_tolerance=0.05,   # 50 ms
        pitch_tolerance=0.25,   # quarter semitone (MIDI pitch comparison)
        offset_ratio=None,      # ignore offset
        offset_min_tolerance=None,
    )
    results["onset_precision"] = float(p)
    results["onset_recall"]    = float(r)
    results["onset_f1"]        = float(f)

    # Tier 2: onset + offset
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=gt_intervals,
        ref_pitches=gt_pitches,
        est_intervals=pred_intervals,
        est_pitches=pred_pitches,
        onset_tolerance=0.05,
        pitch_tolerance=0.25,
        offset_ratio=0.2,       # 20% of note duration
        offset_min_tolerance=0.05,
    )
    results["note_with_offset_precision"] = float(p)
    results["note_with_offset_recall"]    = float(r)
    results["note_with_offset_f1"]        = float(f)

    # Tier 3: onset + offset + velocity (uses separate mir_eval function)
    p, r, f, _ = evaluate_notes_with_velocity(
        ref_intervals=gt_intervals,
        ref_pitches=gt_pitches,
        ref_velocities=gt_vels,
        est_intervals=pred_intervals,
        est_pitches=pred_pitches,
        est_velocities=pred_vels,
        onset_tolerance=0.05,
        pitch_tolerance=0.25,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
        velocity_tolerance=0.1,
    )
    results["note_with_offset_vel_precision"] = float(p)
    results["note_with_offset_vel_recall"]    = float(r)
    results["note_with_offset_vel_f1"]        = float(f)

    return results


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_onset:       torch.Tensor,
    pred_frame:       torch.Tensor,
    pred_offset:      torch.Tensor,
    pred_velocity:    torch.Tensor,
    gt_onset:         torch.Tensor,
    gt_frame:         torch.Tensor,
    gt_offset:        torch.Tensor,
    gt_velocity:      torch.Tensor,
    onset_threshold:  float = 0.5,
    frame_threshold:  float = 0.5,
    offset_threshold: float = 0.5,
    fps:              float = 31.25,
) -> Dict[str, float]:
    """
    Compute all AMT metrics for one file.

    Returns a flat dict containing all frame-level and note-level metrics.
    Safe to call even if mir_eval is not installed — frame metrics are always
    computed; note metrics are skipped with a warning if mir_eval is absent.
    """
    results: Dict[str, float] = {}

    # Frame metrics (no external dependency)
    results.update(compute_frame_metrics(
        pred_frame=pred_frame,
        gt_frame=gt_frame,
        frame_threshold=frame_threshold,
    ))

    # Note metrics (requires mir_eval)
    if _MIR_EVAL_AVAILABLE:
        results.update(compute_note_metrics(
            pred_onset=pred_onset,
            pred_frame=pred_frame,
            pred_offset=pred_offset,
            pred_velocity=pred_velocity,
            gt_onset=gt_onset,
            gt_frame=gt_frame,
            gt_offset=gt_offset,
            gt_velocity=gt_velocity,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            fps=fps,
        ))
    else:
        print("WARNING: mir_eval not installed — note-level metrics skipped.")

    return results
