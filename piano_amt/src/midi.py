"""
midi.py — MIDI loading and piano-roll label encoding/decoding.

Design follows Hawthorne et al. 2018a §3.1 "Label Design":
  - onset_roll:    1.0 for ONSET_WINDOW_FRAMES around each note onset.
  - frame_roll:    1.0 for every frame the note is sounding.
  - offset_roll:   1.0 for OFFSET_WINDOW_FRAMES around each note offset.
                   (jongwook/onsets-and-frames offset head improvement.)
  - velocity_roll: velocity/128 at the onset frame only.

All rolls have shape (n_frames, 88) where dim-1 indexes piano keys [21..108].

Papers:
  Hawthorne et al. 2018a §3.1 — label encoding scheme.
  jongwook/onsets-and-frames — offset head improvement.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import pretty_midi
import torch

from .constants import (
    FRAMES_PER_SECOND,
    MIN_MIDI,
    MAX_MIDI,
    N_KEYS,
    ONSET_WINDOW_FRAMES,
    OFFSET_WINDOW_FRAMES,
    VELOCITY_SCALE,
)

# Type alias for a single note event
NoteEvent = Tuple[float, float, int, int]  # (onset_sec, offset_sec, pitch, velocity)


# ---------------------------------------------------------------------------
# MIDI I/O
# ---------------------------------------------------------------------------

def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """
    Load a MIDI file from disk.

    Args:
        path: Path to .midi or .mid file.

    Returns:
        pretty_midi.PrettyMIDI object.
    """
    return pretty_midi.PrettyMIDI(str(path))


def midi_to_note_events(pm: pretty_midi.PrettyMIDI) -> List[NoteEvent]:
    """
    Extract piano note events from a PrettyMIDI object.

    Filters to the standard 88-key range [MIN_MIDI=21, MAX_MIDI=108].
    All notes from all instruments are merged and sorted by onset time.

    Args:
        pm: Loaded PrettyMIDI object.

    Returns:
        List of (onset_sec, offset_sec, pitch, velocity) tuples sorted by onset.
        pitch is in MIDI note number [21, 108].
        velocity is raw MIDI velocity [0, 127].

    Paper: Hawthorne 2018a §3.1 — piano range filtering.
    """
    events: List[NoteEvent] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            if MIN_MIDI <= note.pitch <= MAX_MIDI:
                events.append((
                    float(note.start),
                    float(note.end),
                    int(note.pitch),
                    int(note.velocity),
                ))
    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
# Label roll encoding
# ---------------------------------------------------------------------------

def note_events_to_rolls(
    events: List[NoteEvent],
    n_frames: int,
    fps: float = FRAMES_PER_SECOND,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a list of note events into four piano-roll tensors.

    Encoding scheme (Hawthorne 2018a §3.1):
      onset_roll[f : f+ONSET_WIN, key]      = 1.0  (frames at/around onset)
      frame_roll[onset_f : offset_f+1, key] = 1.0  (all sounding frames)
      offset_roll[off_f : off_f+OFF_WIN, key]= 1.0 (frames at/around offset)
      velocity_roll[onset_f, key]            = velocity / 128.0

    Args:
        events:   List of (onset_sec, offset_sec, pitch, velocity) tuples.
        n_frames: Total number of frames in the output roll.
        fps:      Frames per second (default FRAMES_PER_SECOND = 31.25).

    Returns:
        onset_roll:    FloatTensor (n_frames, 88)
        frame_roll:    FloatTensor (n_frames, 88)
        offset_roll:   FloatTensor (n_frames, 88)
        velocity_roll: FloatTensor (n_frames, 88)

    Shape:
        All outputs: (n_frames, N_KEYS) = (n_frames, 88)

    Papers:
        Hawthorne 2018a §3.1: onset/frame/velocity encoding.
        jongwook/onsets-and-frames: offset head improvement.
    """
    onset_roll    = torch.zeros(n_frames, N_KEYS, dtype=torch.float32)
    frame_roll    = torch.zeros(n_frames, N_KEYS, dtype=torch.float32)
    offset_roll   = torch.zeros(n_frames, N_KEYS, dtype=torch.float32)
    velocity_roll = torch.zeros(n_frames, N_KEYS, dtype=torch.float32)

    for onset_sec, offset_sec, pitch, velocity in events:
        key = pitch - MIN_MIDI          # [0, 87]
        if key < 0 or key >= N_KEYS:
            continue

        onset_f  = int(math.floor(onset_sec  * fps))
        offset_f = int(math.floor(offset_sec * fps))

        # Clamp to valid frame range
        onset_f  = max(0, min(onset_f,  n_frames - 1))
        offset_f = max(0, min(offset_f, n_frames - 1))

        # Onset window: [onset_f, onset_f + ONSET_WINDOW_FRAMES)
        on_start = onset_f
        on_end   = min(onset_f + ONSET_WINDOW_FRAMES, n_frames)
        onset_roll[on_start:on_end, key] = 1.0

        # Frame roll: [onset_f, offset_f + 1)
        fr_start = onset_f
        fr_end   = min(offset_f + 1, n_frames)
        if fr_start < fr_end:
            frame_roll[fr_start:fr_end, key] = 1.0

        # Offset window: [offset_f, offset_f + OFFSET_WINDOW_FRAMES)
        off_start = offset_f
        off_end   = min(offset_f + OFFSET_WINDOW_FRAMES, n_frames)
        offset_roll[off_start:off_end, key] = 1.0

        # Velocity at onset frame only
        velocity_roll[onset_f, key] = velocity / VELOCITY_SCALE

    return onset_roll, frame_roll, offset_roll, velocity_roll


def midi_path_to_rolls(
    midi_path: str | Path,
    n_frames: int,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a MIDI file and produce windowed piano-roll tensors.

    Clips note events to [start_sec, start_sec + duration_sec], then
    re-times all events so they begin at t=0 relative to the window.

    Args:
        midi_path:    Path to MIDI file.
        n_frames:     Number of output frames (height of rolls).
        start_sec:    Window start in seconds (default 0 = whole file).
        duration_sec: Window duration in seconds.  If None, uses entire file.

    Returns:
        (onset_roll, frame_roll, offset_roll, velocity_roll)
        Each has shape (n_frames, 88).

    Shape:
        All outputs: (n_frames, 88)
    """
    pm = load_midi(midi_path)
    events = midi_to_note_events(pm)

    if duration_sec is None:
        end_sec = pm.get_end_time()
    else:
        end_sec = start_sec + duration_sec

    # Clip events to window [start_sec, end_sec], shift to start at 0
    windowed: List[NoteEvent] = []
    for onset_sec, offset_sec, pitch, velocity in events:
        if offset_sec <= start_sec or onset_sec >= end_sec:
            continue  # outside window entirely
        # Clip onset/offset to window boundaries and shift origin
        clipped_onset  = max(onset_sec,  start_sec) - start_sec
        clipped_offset = min(offset_sec, end_sec)   - start_sec
        windowed.append((clipped_onset, clipped_offset, pitch, velocity))

    return note_events_to_rolls(windowed, n_frames, fps=FRAMES_PER_SECOND)


# ---------------------------------------------------------------------------
# Post-processing decoder: rolls → pretty_midi
# ---------------------------------------------------------------------------

def rolls_to_midi(
    onset_roll:     torch.Tensor,
    frame_roll:     torch.Tensor,
    velocity_roll:  torch.Tensor,
    fps:            float = FRAMES_PER_SECOND,
    onset_threshold:  float = 0.5,
    frame_threshold:  float = 0.5,
) -> pretty_midi.PrettyMIDI:
    """
    Decode piano-roll tensors back to a PrettyMIDI object.

    Algorithm (Hawthorne 2018a §4 post-processing):
      1. A note starts at frame f when onset_roll[f, key] > onset_threshold.
      2. Its duration extends while frame_roll[f:, key] > frame_threshold.
      3. Open notes at end-of-clip are closed at the last frame.
      4. Minimum note duration is 16 ms (prevent sub-frame artefacts).

    Args:
        onset_roll:       Tensor (T, 88), onset probabilities or binary labels.
        frame_roll:       Tensor (T, 88), frame probabilities or binary labels.
        velocity_roll:    Tensor (T, 88), velocity values in [0, 1].
        fps:              Frames per second (default FRAMES_PER_SECOND = 31.25).
        onset_threshold:  Threshold for onset detection.
        frame_threshold:  Threshold for frame activation.

    Returns:
        pretty_midi.PrettyMIDI with a single piano instrument containing
        all decoded notes.

    Shape:
        onset_roll, frame_roll, velocity_roll: (T, 88)
    """
    MIN_NOTE_DURATION_SEC = 0.016  # 16 ms minimum

    pm   = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # acoustic grand piano

    onset_np    = (onset_roll > onset_threshold).cpu().numpy()
    frame_np    = (frame_roll > frame_threshold).cpu().numpy()
    velocity_np = velocity_roll.cpu().numpy()

    T = onset_np.shape[0]

    for key in range(N_KEYS):
        pitch       = key + MIN_MIDI
        note_start  = None  # frame index of active note start
        note_vel    = 64    # default velocity

        for f in range(T):
            is_onset = onset_np[f, key]
            is_active = frame_np[f, key]

            if is_onset:
                # If a previous note is still open, close it first
                if note_start is not None:
                    end_sec   = f / fps
                    start_sec = note_start / fps
                    if end_sec - start_sec >= MIN_NOTE_DURATION_SEC:
                        piano.notes.append(
                            pretty_midi.Note(
                                velocity=note_vel,
                                pitch=pitch,
                                start=start_sec,
                                end=end_sec,
                            )
                        )
                note_start = f
                raw_vel    = velocity_np[f, key]
                note_vel   = max(1, min(127, int(raw_vel * VELOCITY_SCALE)))

            elif note_start is not None and not is_active:
                # Note ended (no longer active and no new onset)
                end_sec   = f / fps
                start_sec = note_start / fps
                if end_sec - start_sec >= MIN_NOTE_DURATION_SEC:
                    piano.notes.append(
                        pretty_midi.Note(
                            velocity=note_vel,
                            pitch=pitch,
                            start=start_sec,
                            end=end_sec,
                        )
                    )
                note_start = None

        # Close any open note at end of clip
        if note_start is not None:
            end_sec   = T / fps
            start_sec = note_start / fps
            if end_sec - start_sec >= MIN_NOTE_DURATION_SEC:
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=note_vel,
                        pitch=pitch,
                        start=start_sec,
                        end=end_sec,
                    )
                )

    piano.notes.sort(key=lambda n: n.start)
    pm.instruments.append(piano)
    return pm
