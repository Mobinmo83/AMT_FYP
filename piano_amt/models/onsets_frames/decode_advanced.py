"""
models/onsets_frames/decode_advanced.py — Advanced piano-roll → note-event decoding
with post-processing methods.

This module extends the original decode.py with 7 post-processing methods
that can be toggled independently or combined for ablation studies.

All methods preserve the original decode.py interface — the original
rolls_to_note_events() and rolls_to_midi_file() remain untouched.

Post-processing methods (ordered by recommended priority):
  1. Onset-Conditioned Offset Estimation  (modifies decoding logic)
  2. Frame-Level Smoothing                (pre-processes frame roll)
  3. Minimum Note Duration Enforcement    (adjustable MIN_DUR)
  4. Velocity-Aware Duplicate Removal     (post-decode event filter)
  5. Chord-Aware Onset Grouping           (post-decode event adjustment)
  6. Adaptive Thresholding                (per-piece threshold tuning)
  7. Sustain Pedal-Aware Offset Extension (extends offsets in pedal regions)

Usage:
    from models.onsets_frames.decode_advanced import advanced_rolls_to_note_events

    events = advanced_rolls_to_note_events(
        onset_roll=pred_onset,
        frame_roll=pred_frame,
        offset_roll=pred_offset,
        velocity_roll=pred_velocity,
        fps=31.25,
        onset_threshold=0.5,
        frame_threshold=0.5,
        # Post-processing toggles:
        use_onset_conditioned_offset=True,   # Method 1
        use_frame_smoothing=True,            # Method 2
        min_note_duration_ms=50.0,           # Method 3 (default was 16ms)
        use_duplicate_removal=True,          # Method 4
        use_chord_grouping=True,             # Method 5
        use_adaptive_thresholds=False,       # Method 6
        use_pedal_extension=False,           # Method 7
    )

Papers:
  Hawthorne et al. 2018a §4 — base decoding algorithm.
  jongwook/onsets-and-frames — offset head improvements.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Dict

import numpy as np
import torch

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.constants import FRAMES_PER_SECOND, MIN_MIDI, N_KEYS, VELOCITY_SCALE

# Import original NoteEvent for compatibility
from models.onsets_frames.decode import NoteEvent


# ---------------------------------------------------------------------------
# Method 2: Frame-Level Smoothing
# ---------------------------------------------------------------------------

def smooth_frame_roll(
    frame_roll: torch.Tensor,  # (T, 88)
    kernel_size: int = 7,
    method: str = "median",
) -> torch.Tensor:
    """
    Smooth the frame activation roll to stabilise flickering activations.

    This reduces premature note-offs and fragmented notes by removing
    short-duration gaps and spikes in the frame activation.

    Args:
        frame_roll: (T, 88) frame probabilities in [0,1].
        kernel_size: Size of the smoothing window (must be odd).
        method: "median" for median filter, "closing" for morphological closing.

    Returns:
        Smoothed frame roll (T, 88).
    """
    frame_np = frame_roll.cpu().numpy()
    T, K = frame_np.shape
    smoothed = np.copy(frame_np)

    if method == "median":
        # Apply median filter along the time axis for each key
        half = kernel_size // 2
        for key in range(K):
            col = frame_np[:, key]
            padded = np.pad(col, (half, half), mode='edge')
            for t in range(T):
                smoothed[t, key] = np.median(padded[t:t + kernel_size])

    elif method == "closing":
        # Morphological closing: dilate then erode (fills small gaps)
        half = kernel_size // 2
        for key in range(K):
            col = (frame_np[:, key] > 0.5).astype(float)
            # Dilate
            dilated = np.copy(col)
            for t in range(T):
                start = max(0, t - half)
                end = min(T, t + half + 1)
                if col[start:end].max() > 0.5:
                    dilated[t] = 1.0
            # Erode
            eroded = np.copy(dilated)
            for t in range(T):
                start = max(0, t - half)
                end = min(T, t + half + 1)
                if dilated[start:end].min() < 0.5:
                    eroded[t] = 0.0
            # Blend: use eroded binary as mask, keep original probabilities
            smoothed[:, key] = frame_np[:, key] * eroded + \
                               frame_np[:, key] * 0.5 * (1 - eroded)

    return torch.from_numpy(smoothed).float()


# ---------------------------------------------------------------------------
# Method 6: Adaptive Thresholding
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(
    onset_roll: torch.Tensor,   # (T, 88)
    frame_roll: torch.Tensor,   # (T, 88)
    onset_base: float = 0.5,
    frame_base: float = 0.5,
    onset_k: float = 0.0,
    frame_k: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute per-piece adaptive thresholds based on activation statistics.

    The threshold is: base + k * std(activations)

    When k=0, this returns the base thresholds (no adaptation).

    Args:
        onset_roll: (T, 88) onset probabilities.
        frame_roll: (T, 88) frame probabilities.
        onset_base: Base onset threshold.
        frame_base: Base frame threshold.
        onset_k: Multiplier for onset std deviation.
        frame_k: Multiplier for frame std deviation.

    Returns:
        (onset_threshold, frame_threshold) — adapted for this piece.
    """
    onset_np = onset_roll.cpu().numpy()
    frame_np = frame_roll.cpu().numpy()

    onset_std = onset_np.std()
    frame_std = frame_np.std()

    onset_thresh = onset_base + onset_k * onset_std
    frame_thresh = frame_base + frame_k * frame_std

    # Clamp to reasonable range
    onset_thresh = max(0.1, min(0.9, onset_thresh))
    frame_thresh = max(0.1, min(0.9, frame_thresh))

    return onset_thresh, frame_thresh


# ---------------------------------------------------------------------------
# Advanced decoding with Method 1 (Onset-Conditioned Offset) and Method 3
# ---------------------------------------------------------------------------

def advanced_decode_notes(
    onset_roll: torch.Tensor,      # (T, 88)
    frame_roll: torch.Tensor,      # (T, 88)
    offset_roll: torch.Tensor,     # (T, 88)
    velocity_roll: torch.Tensor,   # (T, 88)
    fps: float = FRAMES_PER_SECOND,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    min_note_duration_ms: float = 16.0,
    use_onset_conditioned_offset: bool = False,
) -> List[NoteEvent]:
    """
    Decode piano-roll tensors into note events with advanced offset handling.

    When use_onset_conditioned_offset=True (Method 1):
      A note ends when EITHER:
        (a) the frame activation drops below frame_threshold, OR
        (b) the offset head fires above offset_threshold
      whichever comes first after the onset. This uses the offset head
      that jongwook added but the original decode.py doesn't fully exploit.

    When use_onset_conditioned_offset=False:
      Falls back to the standard Hawthorne 2018a algorithm (same as
      original decode.py but with adjustable min_note_duration_ms).

    Args:
        onset_roll:     (T, 88) onset probabilities.
        frame_roll:     (T, 88) frame probabilities.
        offset_roll:    (T, 88) offset probabilities.
        velocity_roll:  (T, 88) velocity values in [0,1].
        fps:            Frames per second.
        onset_threshold:  Threshold for onset detection.
        frame_threshold:  Threshold for frame activation.
        offset_threshold: Threshold for offset detection (Method 1 only).
        min_note_duration_ms: Minimum note duration in milliseconds (Method 3).
        use_onset_conditioned_offset: Enable Method 1.

    Returns:
        List of NoteEvent sorted by onset time.
    """
    MIN_DUR = min_note_duration_ms / 1000.0  # Convert ms to seconds

    onset_np = (onset_roll > onset_threshold).cpu().numpy()
    frame_np = (frame_roll > frame_threshold).cpu().numpy()
    offset_np = (offset_roll > offset_threshold).cpu().numpy() if use_onset_conditioned_offset else None
    velocity_np = velocity_roll.cpu().numpy()

    T = onset_np.shape[0]
    events: List[NoteEvent] = []

    for key in range(N_KEYS):
        pitch = key + MIN_MIDI
        note_start = None
        note_vel = 64

        for f in range(T):
            if onset_np[f, key]:
                # Close previous note if open
                if note_start is not None:
                    dur = (f - note_start) / fps
                    if dur >= MIN_DUR:
                        events.append(NoteEvent(
                            onset_sec=note_start / fps,
                            offset_sec=f / fps,
                            pitch=pitch,
                            velocity=note_vel,
                        ))
                note_start = f
                raw_vel = float(velocity_np[f, key])
                note_vel = max(1, min(127, int(raw_vel * VELOCITY_SCALE)))

            elif note_start is not None:
                # Check for note end conditions
                should_end = False

                if not frame_np[f, key]:
                    # Standard condition: frame dropped below threshold
                    should_end = True

                if use_onset_conditioned_offset and offset_np is not None:
                    if offset_np[f, key]:
                        # Method 1: offset head fired
                        should_end = True

                if should_end:
                    dur = (f - note_start) / fps
                    if dur >= MIN_DUR:
                        events.append(NoteEvent(
                            onset_sec=note_start / fps,
                            offset_sec=f / fps,
                            pitch=pitch,
                            velocity=note_vel,
                        ))
                    note_start = None

        # Close open note at end of piece
        if note_start is not None:
            dur = (T - note_start) / fps
            if dur >= MIN_DUR:
                events.append(NoteEvent(
                    onset_sec=note_start / fps,
                    offset_sec=T / fps,
                    pitch=pitch,
                    velocity=note_vel,
                ))

    events.sort(key=lambda e: e.onset_sec)
    return events


# ---------------------------------------------------------------------------
# Method 4: Velocity-Aware Duplicate Removal
# ---------------------------------------------------------------------------

def remove_duplicate_notes(
    events: List[NoteEvent],
    tolerance_sec: float = 0.05,
) -> List[NoteEvent]:
    """
    Remove duplicate notes: when two onsets on the same pitch occur
    within tolerance_sec, keep only the one with higher velocity.

    Args:
        events: List of NoteEvent (must be sorted by onset time).
        tolerance_sec: Time window for duplicate detection (default 50ms).

    Returns:
        Filtered list of NoteEvent.
    """
    if not events:
        return events

    # Group events by pitch
    by_pitch: Dict[int, List[NoteEvent]] = {}
    for e in events:
        by_pitch.setdefault(e.pitch, []).append(e)

    cleaned: List[NoteEvent] = []

    for pitch, pitch_events in by_pitch.items():
        # Sort by onset time within each pitch
        pitch_events.sort(key=lambda e: e.onset_sec)

        i = 0
        while i < len(pitch_events):
            # Collect all events within tolerance of current
            group = [pitch_events[i]]
            j = i + 1
            while j < len(pitch_events) and \
                  (pitch_events[j].onset_sec - pitch_events[i].onset_sec) <= tolerance_sec:
                group.append(pitch_events[j])
                j += 1


            # Keep onset and velocity from the loudest detection
            best = max(group, key=lambda e: e.velocity)
            # Take the longest offset from the group (recovers offset F1)
            max_group_offset = max(e.offset_sec for e in group)
            # Safety cap: don't extend more than 200ms past the best candidate's
            # own offset — protects against dedup window catching a separate note
            safety_cap = best.offset_sec + 0.200
            merged_offset = min(max_group_offset, safety_cap)
            cleaned.append(best._replace(offset_sec=merged_offset))

            i = j  # Skip past the group

    cleaned.sort(key=lambda e: e.onset_sec)
    return cleaned


# ---------------------------------------------------------------------------
# Method 5: Chord-Aware Onset Grouping
# ---------------------------------------------------------------------------

def group_chord_onsets(
    events: List[NoteEvent],
    tolerance_sec: float = 0.03,
    snap_to: str = "median",
) -> List[NoteEvent]:
    """
    Snap near-simultaneous onsets to a common time to improve chord coherence.

    Groups onsets within tolerance_sec and aligns them to the median
    (or earliest) onset time in the group.

    Args:
        events: List of NoteEvent (sorted by onset time).
        tolerance_sec: Window for grouping simultaneous onsets (default 30ms).
        snap_to: "median" to snap to median onset, "first" to snap to earliest.

    Returns:
        List of NoteEvent with adjusted onset times.
    """
    if not events:
        return events

    # Sort by onset
    sorted_events = sorted(events, key=lambda e: e.onset_sec)

    # Group into chord clusters
    groups: List[List[int]] = []  # indices into sorted_events
    current_group = [0]

    for i in range(1, len(sorted_events)):
        if sorted_events[i].onset_sec - sorted_events[current_group[0]].onset_sec <= tolerance_sec:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    # Snap onsets within each group
    adjusted: List[NoteEvent] = []
    for group in groups:
        if len(group) == 1:
            adjusted.append(sorted_events[group[0]])
            continue

        # Compute target onset time
        onset_times = [sorted_events[idx].onset_sec for idx in group]
        if snap_to == "median":
            target = float(np.median(onset_times))
        elif snap_to == "first":
            target = min(onset_times)
        else:
            target = float(np.median(onset_times))

        for idx in group:
            e = sorted_events[idx]
            # Adjust onset, keep duration the same
            duration = e.offset_sec - e.onset_sec
            adjusted.append(NoteEvent(
                onset_sec=target,
                offset_sec=target + duration,
                pitch=e.pitch,
                velocity=e.velocity,
            ))

    adjusted.sort(key=lambda e: e.onset_sec)
    return adjusted


# ---------------------------------------------------------------------------
# Method 7: Sustain Pedal-Aware Offset Extension
# ---------------------------------------------------------------------------

def extend_offsets_for_pedal(
    events: List[NoteEvent],
    frame_roll: torch.Tensor,  # (T, 88)
    fps: float = FRAMES_PER_SECOND,
    frame_threshold: float = 0.5,
    pedal_energy_threshold: float = 10.0,
    max_extension_sec: float = 2.0,
) -> List[NoteEvent]:
    """
    Extend note offsets in regions where sustain pedal is likely active.

    Heuristic: when many pitches have simultaneous frame activation
    (indicating pedal sustain), extend note offsets to follow the
    actual frame energy decay rather than the initial offset.

    Args:
        events: List of NoteEvent.
        frame_roll: (T, 88) frame probabilities.
        fps: Frames per second.
        frame_threshold: Threshold for frame activity.
        pedal_energy_threshold: Minimum number of simultaneously active
            pitches to consider a region as "pedaled".
        max_extension_sec: Maximum offset extension in seconds.

    Returns:
        List of NoteEvent with potentially extended offsets.
    """
    frame_np = frame_roll.cpu().numpy()
    T, K = frame_np.shape

    # Compute per-frame activity count
    activity = (frame_np > frame_threshold).sum(axis=1)  # (T,)

    # Identify "pedaled" frames
    pedaled = activity >= pedal_energy_threshold  # (T,) bool

    extended: List[NoteEvent] = []
    for e in events:
        offset_frame = int(e.offset_sec * fps)
        key = e.pitch - MIN_MIDI

        if 0 <= key < K and 0 <= offset_frame < T and pedaled[offset_frame]:
            # Extend offset: follow frame activation until it drops
            max_ext_frames = int(max_extension_sec * fps)
            new_offset_frame = offset_frame
            for f in range(offset_frame, min(T, offset_frame + max_ext_frames)):
                if frame_np[f, key] > frame_threshold * 0.5:  # relaxed threshold
                    new_offset_frame = f
                else:
                    break

            new_offset_sec = new_offset_frame / fps
            extended.append(NoteEvent(
                onset_sec=e.onset_sec,
                offset_sec=new_offset_sec,
                pitch=e.pitch,
                velocity=e.velocity,
            ))
        else:
            extended.append(e)

    return extended


# ---------------------------------------------------------------------------
# Main advanced decode function — combines all methods
# ---------------------------------------------------------------------------

def advanced_rolls_to_note_events(
    onset_roll: torch.Tensor,
    frame_roll: torch.Tensor,
    offset_roll: torch.Tensor,
    velocity_roll: torch.Tensor,
    fps: float = FRAMES_PER_SECOND,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    # Method toggles:
    use_onset_conditioned_offset: bool = False,   # Method 1
    use_frame_smoothing: bool = False,            # Method 2
    frame_smoothing_kernel: int = 7,
    frame_smoothing_method: str = "median",
    min_note_duration_ms: float = 16.0,           # Method 3 (raise to 32-50)
    use_duplicate_removal: bool = False,           # Method 4
    duplicate_tolerance_sec: float = 0.05,
    use_chord_grouping: bool = False,              # Method 5
    chord_tolerance_sec: float = 0.03,
    chord_snap_to: str = "median",
    use_adaptive_thresholds: bool = False,         # Method 6
    adaptive_onset_k: float = 0.5,
    adaptive_frame_k: float = 0.5,
    use_pedal_extension: bool = False,             # Method 7
    pedal_energy_threshold: float = 10.0,
    pedal_max_extension_sec: float = 2.0,
) -> List[NoteEvent]:
    """
    Advanced piano-roll to note-event decoder with all post-processing methods.

    Each method can be toggled independently for ablation studies.
    Methods are applied in this order:
      1. Adaptive thresholds (Method 6) — modifies thresholds
      2. Frame smoothing (Method 2) — pre-processes frame roll
      3. Core decoding with onset-conditioned offset (Methods 1 & 3)
      4. Duplicate removal (Method 4)
      5. Chord onset grouping (Method 5)
      6. Pedal offset extension (Method 7)

    Returns:
        List of NoteEvent sorted by onset time.
    """

    # --- Method 6: Adaptive Thresholding ---
    if use_adaptive_thresholds:
        onset_threshold, frame_threshold = compute_adaptive_thresholds(
            onset_roll=onset_roll,
            frame_roll=frame_roll,
            onset_base=onset_threshold,
            frame_base=frame_threshold,
            onset_k=adaptive_onset_k,
            frame_k=adaptive_frame_k,
        )

    # --- Method 2: Frame-Level Smoothing ---
    if use_frame_smoothing:
        frame_roll = smooth_frame_roll(
            frame_roll=frame_roll,
            kernel_size=frame_smoothing_kernel,
            method=frame_smoothing_method,
        )

    # --- Core decoding with Methods 1 & 3 ---
    events = advanced_decode_notes(
        onset_roll=onset_roll,
        frame_roll=frame_roll,
        offset_roll=offset_roll,
        velocity_roll=velocity_roll,
        fps=fps,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        offset_threshold=offset_threshold,
        min_note_duration_ms=min_note_duration_ms,
        use_onset_conditioned_offset=use_onset_conditioned_offset,
    )

    # --- Method 4: Velocity-Aware Duplicate Removal ---
    if use_duplicate_removal:
        events = remove_duplicate_notes(
            events=events,
            tolerance_sec=duplicate_tolerance_sec,
        )

    # --- Method 5: Chord-Aware Onset Grouping ---
    if use_chord_grouping:
        events = group_chord_onsets(
            events=events,
            tolerance_sec=chord_tolerance_sec,
            snap_to=chord_snap_to,
        )

    # --- Method 7: Sustain Pedal-Aware Offset Extension ---
    if use_pedal_extension:
        events = extend_offsets_for_pedal(
            events=events,
            frame_roll=frame_roll,
            fps=fps,
            frame_threshold=frame_threshold,
            pedal_energy_threshold=pedal_energy_threshold,
            max_extension_sec=pedal_max_extension_sec,
        )

    return events


# ---------------------------------------------------------------------------
# Advanced MIDI file output
# ---------------------------------------------------------------------------

def advanced_rolls_to_midi_file(
    onset_roll: torch.Tensor,
    frame_roll: torch.Tensor,
    offset_roll: torch.Tensor,
    velocity_roll: torch.Tensor,
    output_path: str | Path,
    fps: float = FRAMES_PER_SECOND,
    **kwargs,
) -> None:
    """
    Decode piano rolls with advanced post-processing and save as MIDI.

    Accepts all keyword arguments from advanced_rolls_to_note_events().
    """
    import pretty_midi

    events = advanced_rolls_to_note_events(
        onset_roll=onset_roll,
        frame_roll=frame_roll,
        offset_roll=offset_roll,
        velocity_roll=velocity_roll,
        fps=fps,
        **kwargs,
    )

    # Build PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for e in events:
        note = pretty_midi.Note(
            velocity=e.velocity,
            pitch=e.pitch,
            start=e.onset_sec,
            end=e.offset_sec,
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(str(output_path))
    print(f"Advanced MIDI saved → {output_path}  ({len(events)} notes)")
