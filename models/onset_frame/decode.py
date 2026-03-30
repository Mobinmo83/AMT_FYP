"""
models/onsets_frames/decode.py — Piano-roll → note-event decoding.

Provides two public functions:
  rolls_to_note_events() — rolls → List[NoteEvent] (used by evaluate/metrics.py)
  rolls_to_midi_file()   — rolls → save .mid to disk

The decoding algorithm follows Hawthorne 2018a §4:
  A note starts when onset_roll[f, key] > onset_threshold.
  It ends when frame_roll[f, key] drops below frame_threshold.
  Open notes at end-of-clip are closed at the last frame.
  Minimum note duration = 16 ms (1 frame at 31.25 fps ≈ 32 ms → 16 ms guard).

This file wraps src/midi.py rolls_to_midi() and adds the note-event list
extraction needed by mir_eval.

NoteEvent format (mir_eval compatible):
  (onset_sec, offset_sec, pitch_midi, velocity_int)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import torch

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.constants import FRAMES_PER_SECOND, MIN_MIDI, N_KEYS, VELOCITY_SCALE
from src.midi import rolls_to_midi   # reuse existing decoder


# ---------------------------------------------------------------------------
# NoteEvent type
# ---------------------------------------------------------------------------

class NoteEvent(NamedTuple):
    onset_sec:  float
    offset_sec: float
    pitch:      int    # MIDI note number [21..108]
    velocity:   int    # [1..127]


# ---------------------------------------------------------------------------
# rolls_to_note_events
# ---------------------------------------------------------------------------

def rolls_to_note_events(
    onset_roll:       torch.Tensor,
    frame_roll:       torch.Tensor,
    velocity_roll:    torch.Tensor,
    fps:              float = FRAMES_PER_SECOND,
    onset_threshold:  float = 0.5,
    frame_threshold:  float = 0.5,
) -> List[NoteEvent]:
    """
    Decode piano-roll tensors into a list of NoteEvent objects.

    Used by evaluate/metrics.py to interface with mir_eval.

    Args:
        onset_roll:      (T, 88) — onset probabilities in [0,1].
        frame_roll:      (T, 88) — frame probabilities in [0,1].
        velocity_roll:   (T, 88) — velocity values in [0,1].
        fps:             Frames per second (31.25).
        onset_threshold: Threshold for onset detection.
        frame_threshold: Threshold for frame activation.

    Returns:
        List of NoteEvent sorted by onset time.

    Algorithm: Hawthorne 2018a §4 post-processing.
    """
    MIN_DUR = 0.016   # 16 ms

    onset_np    = (onset_roll  > onset_threshold).cpu().numpy()
    frame_np    = (frame_roll  > frame_threshold).cpu().numpy()
    velocity_np = velocity_roll.cpu().numpy()

    T = onset_np.shape[0]
    events: List[NoteEvent] = []

    for key in range(N_KEYS):
        pitch      = key + MIN_MIDI
        note_start = None
        note_vel   = 64

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
                raw_vel  = float(velocity_np[f, key])
                note_vel = max(1, min(127, int(raw_vel * VELOCITY_SCALE)))

            elif note_start is not None and not frame_np[f, key]:
                dur = (f - note_start) / fps
                if dur >= MIN_DUR:
                    events.append(NoteEvent(
                        onset_sec=note_start / fps,
                        offset_sec=f / fps,
                        pitch=pitch,
                        velocity=note_vel,
                    ))
                note_start = None

        # Close open note at end
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
# rolls_to_midi_file
# ---------------------------------------------------------------------------

def rolls_to_midi_file(
    onset_roll:       torch.Tensor,
    frame_roll:       torch.Tensor,
    velocity_roll:    torch.Tensor,
    output_path:      str | Path,
    fps:              float = FRAMES_PER_SECOND,
    onset_threshold:  float = 0.5,
    frame_threshold:  float = 0.5,
) -> None:
    """
    Decode piano rolls and save as a MIDI file.

    Args:
        onset_roll:    (T, 88) — onset probabilities in [0,1].
        frame_roll:    (T, 88) — frame probabilities in [0,1].
        velocity_roll: (T, 88) — velocity values in [0,1].
        output_path:   Path to write .mid file.
        fps:           Frames per second (31.25).
        onset_threshold:  Threshold for onset detection.
        frame_threshold:  Threshold for frame activation.
    """
    pm = rolls_to_midi(
        onset_roll=onset_roll,
        frame_roll=frame_roll,
        velocity_roll=velocity_roll,
        fps=fps,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    pm.write(str(output_path))
    print(f"MIDI saved → {output_path}  ({len(pm.instruments[0].notes)} notes)")
