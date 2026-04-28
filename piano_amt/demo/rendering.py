from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import soundfile as sf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

import contextlib
import io
import warnings
from IPython.display import HTML, display

from demo.demo_config import DEFAULT_SF2_PATHS, SAMPLE_RATE
from demo.inference import DemoNoteEvent
from src.constants import FRAMES_PER_SECOND


# ---------------------------------------------------------------------------
# Public demo constants
# ---------------------------------------------------------------------------

DEMO_FPS = float(FRAMES_PER_SECOND)
PIANO_LOW = 21
PIANO_HIGH = 108

PIANO_PROGRAMS = {
    "Acoustic Grand": 0,
    "Bright Acoustic": 1,
    "Electric Grand": 2,
    "Honky-tonk": 3,
    "Electric Piano": 4,
}



def _velocity_to_midi_value(v) -> int:
    v = float(v)
    if v <= 1.0:
        v = v * 127.0
    return int(np.clip(round(v), 1, 127))


def crop_note_events(
    note_events,
    start_sec: float,
    end_sec: float,
    shift_to_zero: bool = True,
):
    """Crop decoded note events to a display window.

    The output keeps the same note-event representation used by the demo.
    If shift_to_zero=True, the cropped excerpt starts at t=0 for clean display.
    """
    cropped = []

    for ev in list(note_events):
        onset, offset, pitch, velocity = _event_to_fields(ev)

        if offset <= start_sec or onset >= end_sec:
            continue

        new_onset = max(onset, start_sec)
        new_offset = min(offset, end_sec)

        if shift_to_zero:
            new_onset -= start_sec
            new_offset -= start_sec

        if new_offset > new_onset:
            cropped.append(
                DemoNoteEvent(
                    int(pitch),
                    float(new_onset),
                    float(new_offset),
                    _velocity_to_midi_value(velocity),
                )
            )

    cropped.sort(key=lambda e: (e.onset_sec, e.pitch, e.offset_sec))
    return cropped


def crop_pretty_midi(
    pm: pretty_midi.PrettyMIDI,
    start_sec: float,
    end_sec: float,
    shift_to_zero: bool = True,
) -> pretty_midi.PrettyMIDI:
    """Crop a PrettyMIDI object to a short excerpt for clean display/playback."""
    out = pretty_midi.PrettyMIDI()

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name,
        )

        for n in inst.notes:
            if n.end <= start_sec or n.start >= end_sec:
                continue

            new_start = max(n.start, start_sec)
            new_end = min(n.end, end_sec)

            if shift_to_zero:
                new_start -= start_sec
                new_end -= start_sec

            if new_end > new_start:
                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(n.velocity),
                        pitch=int(n.pitch),
                        start=float(new_start),
                        end=float(new_end),
                    )
                )

        for cc in inst.control_changes:
            if start_sec <= cc.time < end_sec:
                new_time = cc.time - start_sec if shift_to_zero else cc.time
                new_inst.control_changes.append(
                    pretty_midi.ControlChange(
                        number=int(cc.number),
                        value=int(cc.value),
                        time=float(new_time),
                    )
                )

        if new_inst.notes or new_inst.control_changes:
            out.instruments.append(new_inst)

    return out


def save_pretty_midi(pm: pretty_midi.PrettyMIDI, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    return output_path


def save_audio_excerpt(
    audio_path: str | Path,
    output_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> Path:
    """Save an audio excerpt matching the MIDI display window."""
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(audio_path))
    sr = int(info.samplerate)

    start_frame = max(0, int(round(start_sec * sr)))
    stop_frame = max(start_frame + 1, int(round(end_sec * sr)))
    frames = stop_frame - start_frame

    y, _ = sf.read(str(audio_path), start=start_frame, frames=frames, dtype="float32")
    sf.write(str(output_path), y, sr)

    return output_path


def select_demo_window(
    note_events,
    duration_sec: float = 25.0,
    preferred_start_sec: float | None = None,
    hop_sec: float = 5.0,
):
    """Choose a short, non-empty window for clean demo visualisation.

    If preferred_start_sec is given, it uses that exact window.
    Otherwise, it chooses the window with the most decoded note activity.
    """
    note_events = list(note_events)

    if preferred_start_sec is not None:
        start = max(0.0, float(preferred_start_sec))
        return start, start + float(duration_sec)

    if not note_events:
        return 0.0, float(duration_sec)

    parsed = [_event_to_fields(ev) for ev in note_events]
    max_time = max(offset for _, offset, _, _ in parsed)

    if max_time <= duration_sec:
        return 0.0, max(float(duration_sec), float(max_time))

    best_start = 0.0
    best_score = -1.0

    last_start = max(0.0, max_time - duration_sec)
    candidates = np.arange(0.0, last_start + 1e-6, hop_sec)

    for start in candidates:
        end = start + duration_sec

        score = 0.0
        for onset, offset, _, velocity in parsed:
            overlap = max(0.0, min(offset, end) - max(onset, start))
            if overlap > 0:
                score += 1.0 + 0.25 * min(overlap, 2.0) + 0.002 * _velocity_to_midi_value(velocity)

        if score > best_score:
            best_score = score
            best_start = float(start)

    return best_start, best_start + float(duration_sec)


def plot_event_comparison_bars(
    gt_events,
    pred_events,
    title: str = "Ground truth vs predicted MIDI events",
    save_path: str | Path | None = None,
    figsize=(14, 7),
):
    """Professional two-panel comparison of GT and predicted decoded MIDI events.

    This is clearer than three separate MIDI visualisations.
    """
    gt_events = list(gt_events)
    pred_events = list(pred_events)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    panels = [
        (axes[0], gt_events, "Evaluation ground truth MIDI excerpt", "0.35"),
        (axes[1], pred_events, "Predicted MIDI excerpt", "tab:blue"),
    ]

    all_events = gt_events + pred_events

    if all_events:
        parsed_all = [_event_to_fields(ev) for ev in all_events]
        max_time = max(offset for _, offset, _, _ in parsed_all)
        min_pitch = min(pitch for _, _, pitch, _ in parsed_all)
        max_pitch = max(pitch for _, _, pitch, _ in parsed_all)
    else:
        max_time = 1.0
        min_pitch = 21
        max_pitch = 108

    for ax, events, subtitle, color in panels:
        for ev in events:
            onset, offset, pitch, velocity = _event_to_fields(ev)
            duration = max(0.01, offset - onset)

            rect = Rectangle(
                (onset, pitch - 0.38),
                duration,
                0.76,
                facecolor=color,
                edgecolor="black",
                linewidth=0.12,
                alpha=0.85,
            )
            ax.add_patch(rect)

        ax.set_title(subtitle)
        ax.set_ylabel("MIDI pitch")
        ax.grid(True, axis="x", alpha=0.25)

        ticks, labels = _piano_pitch_ticks()
        valid = [
            (t, lab)
            for t, lab in zip(ticks, labels)
            if max(20, min_pitch - 2) <= t <= min(109, max_pitch + 2)
        ]
        if valid:
            ax.set_yticks([t for t, _ in valid])
            ax.set_yticklabels([lab for _, lab in valid])

    axes[-1].set_xlabel("Time in excerpt (s)")
    axes[0].set_xlim(0, max_time + 0.25)
    axes[0].set_ylim(max(20, min_pitch - 2), min(109, max_pitch + 2))

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")

    return fig


# ---------------------------------------------------------------------------
# Audio / MIDI synthesis helpers
# ---------------------------------------------------------------------------

def clone_pretty_midi(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Clone a PrettyMIDI object through an in-memory temporary MIDI file."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mid") as tmp:
        pm.write(tmp.name)
        tmp.flush()
        return pretty_midi.PrettyMIDI(tmp.name)


def apply_piano_program(
    pm: pretty_midi.PrettyMIDI,
    piano_sound: str = "Acoustic Grand",
) -> pretty_midi.PrettyMIDI:
    """Return a cloned PrettyMIDI object with all non-drum instruments set to a piano program."""
    program = PIANO_PROGRAMS.get(piano_sound, 0)
    pm2 = clone_pretty_midi(pm)

    for inst in pm2.instruments:
        inst.program = int(program)
        inst.is_drum = False

    return pm2


def find_default_sf2() -> str | None:
    """Find the first available default soundfont path configured for the demo."""
    for p in DEFAULT_SF2_PATHS:
        if Path(p).exists():
            return str(p)
    return None


def synthesize_pretty_midi(
    pm: pretty_midi.PrettyMIDI,
    sr: int = SAMPLE_RATE,
    piano_sound: str = "Acoustic Grand",
    sf2_path: str | None = None,
) -> np.ndarray:
    """Synthesize a PrettyMIDI object to audio.

    Uses FluidSynth when available, otherwise falls back to PrettyMIDI synthesis.
    The waveform is peak-normalised for notebook playback.
    """
    pm2 = apply_piano_program(pm, piano_sound)
    sf2_path = sf2_path or find_default_sf2()

    try:
        if sf2_path:
            y = pm2.fluidsynth(fs=sr, sf2_path=sf2_path)
        else:
            y = pm2.fluidsynth(fs=sr)
    except Exception:
        y = pm2.synthesize(fs=sr)

    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        y = np.zeros(sr, dtype=np.float32)

    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs > 0:
        y = y / max_abs

    return y


def save_audio_wav(
    y: np.ndarray,
    output_path: str | Path,
    sr: int = SAMPLE_RATE,
) -> Path:
    """Save a waveform as a WAV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.asarray(y, dtype=np.float32), sr)
    return output_path


def synthesize_and_save(
    pm: pretty_midi.PrettyMIDI,
    output_path: str | Path,
    piano_sound: str = "Acoustic Grand",
    sr: int = SAMPLE_RATE,
) -> Path:
    """Synthesize MIDI to WAV and save it."""
    y = synthesize_pretty_midi(pm, sr=sr, piano_sound=piano_sound)
    return save_audio_wav(y, output_path, sr=sr)


# ---------------------------------------------------------------------------
# Generic conversion helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def midi_to_events(pm: pretty_midi.PrettyMIDI) -> list[DemoNoteEvent]:
    """Convert a PrettyMIDI object into sorted demo note events."""
    events: list[DemoNoteEvent] = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        for n in inst.notes:
            if PIANO_LOW <= int(n.pitch) <= PIANO_HIGH and float(n.end) > float(n.start):
                events.append(
                    DemoNoteEvent(
                        pitch=int(n.pitch),
                        onset_sec=float(n.start),
                        offset_sec=float(n.end),
                        velocity=int(np.clip(n.velocity, 1, 127)),
                    )
                )

    events.sort(key=lambda e: (e.onset_sec, e.pitch, e.offset_sec))
    return events


def _event_to_fields(event) -> tuple[float, float, int, float]:
    """Convert an event into (onset_sec, offset_sec, pitch, velocity).

    Main supported type is DemoNoteEvent, but this also accepts common dict/object/tuple forms.
    """
    if hasattr(event, "onset_sec") and hasattr(event, "offset_sec"):
        onset = float(getattr(event, "onset_sec"))
        offset = float(getattr(event, "offset_sec"))
        pitch = int(getattr(event, "pitch"))
        velocity = float(getattr(event, "velocity", 64))
        return onset, max(offset, onset + 1e-3), pitch, velocity

    if isinstance(event, dict):
        onset = event.get(
            "onset_sec",
            event.get("onset_time", event.get("start_time", event.get("onset", event.get("start")))),
        )
        offset = event.get(
            "offset_sec",
            event.get("offset_time", event.get("end_time", event.get("offset", event.get("end")))),
        )
        pitch = event.get("pitch", event.get("midi_note", event.get("note")))
        velocity = event.get("velocity", 64)

        if onset is None or offset is None or pitch is None:
            raise ValueError(f"Unsupported note-event dictionary format: {event}")

        onset = float(onset)
        offset = float(offset)
        return onset, max(offset, onset + 1e-3), int(pitch), float(velocity)

    if hasattr(event, "onset_time") or hasattr(event, "offset_time"):
        onset = getattr(event, "onset_time", getattr(event, "start_time", getattr(event, "onset", None)))
        offset = getattr(event, "offset_time", getattr(event, "end_time", getattr(event, "offset", None)))
        pitch = getattr(event, "midi_note", getattr(event, "pitch", getattr(event, "note", None)))
        velocity = getattr(event, "velocity", 64)

        if onset is None or offset is None or pitch is None:
            raise ValueError(f"Unsupported note-event object format: {event}")

        onset = float(onset)
        offset = float(offset)
        return onset, max(offset, onset + 1e-3), int(pitch), float(velocity)

    vals = list(event)
    if len(vals) < 4:
        raise ValueError(f"Unsupported note-event format: {event}")

    onset, offset, pitch, velocity = vals[:4]
    onset = float(onset)
    offset = float(offset)
    return onset, max(offset, onset + 1e-3), int(pitch), float(velocity)


def _parse_events(events: Iterable) -> list[tuple[float, float, int, float]]:
    """Parse, filter and sort note events."""
    parsed: list[tuple[float, float, int, float]] = []

    for ev in events:
        onset, offset, pitch, velocity = _event_to_fields(ev)
        if offset <= onset:
            continue
        if pitch < PIANO_LOW or pitch > PIANO_HIGH:
            continue
        parsed.append((onset, offset, pitch, velocity))

    parsed.sort(key=lambda x: (x[0], x[2], x[1]))
    return parsed


def _clip_events_to_window(
    parsed: Sequence[tuple[float, float, int, float]],
    start_time: float,
    end_time: float,
) -> list[tuple[float, float, int, float]]:
    """Clip events to a visible time window."""
    clipped: list[tuple[float, float, int, float]] = []

    for onset, offset, pitch, velocity in parsed:
        if offset < start_time or onset > end_time:
            continue

        onset_c = max(onset, start_time)
        offset_c = min(offset, end_time)

        if offset_c > onset_c:
            clipped.append((onset_c, offset_c, pitch, velocity))

    return clipped


def _resolve_time_window(
    parsed: Sequence[tuple[float, float, int, float]],
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = None,
) -> tuple[float, float]:
    """Resolve plotting time window from explicit or inferred values."""
    if not parsed:
        return 0.0, 1.0

    inferred_start = 0.0
    inferred_end = max(offset for _, offset, _, _ in parsed)

    start = float(start_time) if start_time is not None else inferred_start

    if window_duration is not None:
        end = start + float(window_duration)
    elif end_time is not None:
        end = float(end_time)
    else:
        end = inferred_end

    if end <= start:
        end = start + 1.0

    return start, end


def _piano_pitch_ticks(
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
) -> tuple[list[int], list[str]]:
    """Return pitch tick positions and labels at C notes."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    ticks = [p for p in range(24, 109, 12) if pitch_min <= p <= pitch_max]
    labels = [f"{names[p % 12]}{(p // 12) - 1}" for p in ticks]
    return ticks, labels


def _velocity_norm(parsed: Sequence[tuple[float, float, int, float]]) -> Normalize:
    """Choose a stable velocity normalisation."""
    if not parsed:
        return Normalize(vmin=0.0, vmax=127.0)

    vmax = max(float(v) for _, _, _, v in parsed)

    if vmax <= 1.0:
        return Normalize(vmin=0.0, vmax=1.0)

    return Normalize(vmin=0.0, vmax=127.0)


def _draw_note_bars(
    ax,
    parsed: Sequence[tuple[float, float, int, float]],
    *,
    start_time: float,
    end_time: float,
    color_mode: str = "velocity",
    fixed_color: str = "#4C72B0",
    cmap_name: str = "viridis",
    alpha: float = 0.88,
    bar_height: float = 0.78,
    linewidth: float = 0.15,
    edgecolor: str = "black",
    label: str | None = None,
    velocity_norm: Normalize | None = None,
):
    """Draw horizontal note bars on an existing axis."""
    visible = _clip_events_to_window(parsed, start_time, end_time)

    if not visible:
        return None

    cmap = plt.get_cmap(cmap_name)
    norm = velocity_norm or _velocity_norm(visible)

    for onset, offset, pitch, velocity in visible:
        duration = max(0.005, offset - onset)

        if color_mode == "velocity":
            facecolor = cmap(norm(float(velocity)))
        else:
            facecolor = fixed_color

        rect = Rectangle(
            (onset, pitch - (bar_height / 2.0)),
            duration,
            bar_height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )
        ax.add_patch(rect)

    if color_mode == "velocity":
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return sm

    return None


def _set_pitch_axis(ax, pitch_min: int = PIANO_LOW, pitch_max: int = PIANO_HIGH) -> None:
    ticks, labels = _piano_pitch_ticks(pitch_min, pitch_max)
    ax.set_ylim(pitch_min - 0.5, pitch_max + 0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylabel("MIDI pitch")


# ---------------------------------------------------------------------------
# Professional decoded-MIDI plots
# ---------------------------------------------------------------------------

def plot_midi_event_bars(
    note_events: Iterable,
    title: str = "Decoded MIDI note events",
    save_path: str | Path | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = None,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    figsize=(14, 6),
    cmap_name: str = "viridis",
):
    """Professional MIDI-style note-event plot.

    This should be the main display for predicted MIDI:
    x-axis = time in seconds,
    y-axis = pitch,
    note length = horizontal bar length,
    colour = velocity.
    """
    parsed = _parse_events(note_events)
    start, end = _resolve_time_window(parsed, start_time, end_time, window_duration)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not parsed:
        ax.text(0.5, 0.5, "No note events", ha="center", va="center", transform=ax.transAxes)
    else:
        visible = _clip_events_to_window(parsed, start, end)
        if visible:
            min_pitch = max(pitch_min, min(p for _, _, p, _ in visible) - 2)
            max_pitch = min(pitch_max, max(p for _, _, p, _ in visible) + 2)
        else:
            min_pitch, max_pitch = pitch_min, pitch_max

        sm = _draw_note_bars(
            ax,
            parsed,
            start_time=start,
            end_time=end,
            color_mode="velocity",
            cmap_name=cmap_name,
            velocity_norm=_velocity_norm(parsed),
        )

        if sm is not None:
            cbar = fig.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label("Velocity")

        _set_pitch_axis(ax, min_pitch, max_pitch)

    ax.set_xlim(start, end)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=180, facecolor="white")

    return fig


# Backwards-compatible name used in your notebook.
def plot_note_events_bars(*args, **kwargs):
    return plot_midi_event_bars(*args, **kwargs)


def note_events_to_roll(
    note_events: Iterable,
    *,
    fps: float = DEMO_FPS,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    start_time: float = 0.0,
    end_time: float | None = None,
    use_velocity: bool = True,
) -> np.ndarray:
    """Convert decoded MIDI events into a piano-roll matrix.

    Output shape: (88, n_frames), where rows correspond to MIDI pitches 21--108.
    """
    parsed = _parse_events(note_events)
    n_pitches = pitch_max - pitch_min + 1

    if not parsed:
        return np.zeros((n_pitches, 1), dtype=np.float32)

    if end_time is None:
        end_time = max(offset for _, offset, _, _ in parsed)

    start = float(start_time)
    end = float(end_time)

    if end <= start:
        end = start + 1.0

    n_frames = max(1, int(np.ceil((end - start) * fps)))
    roll = np.zeros((n_pitches, n_frames), dtype=np.float32)

    visible = _clip_events_to_window(parsed, start, end)

    for onset, offset, pitch, velocity in visible:
        if pitch < pitch_min or pitch > pitch_max:
            continue

        start_idx = max(0, int(np.floor((onset - start) * fps)))
        end_idx = max(start_idx + 1, int(np.ceil((offset - start) * fps)))
        end_idx = min(end_idx, n_frames)

        value = float(velocity)
        if use_velocity:
            if value > 1.0:
                value = value / 127.0
        else:
            value = 1.0

        row = pitch - pitch_min
        roll[row, start_idx:end_idx] = np.maximum(roll[row, start_idx:end_idx], value)

    return roll


def plot_decoded_event_roll(
    note_events: Iterable,
    title: str = "Decoded MIDI piano roll",
    save_path: str | Path | None = None,
    *,
    fps: float = DEMO_FPS,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = None,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    figsize=(14, 6),
    cmap: str = "magma",
):
    """Plot a piano roll reconstructed from decoded MIDI events.

    This is the correct main piano-roll view for predicted MIDI, because it reflects
    the actual note events that are written to the MIDI file.
    """
    parsed = _parse_events(note_events)
    start, end = _resolve_time_window(parsed, start_time, end_time, window_duration)

    roll = note_events_to_roll(
        parsed,
        fps=fps,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        start_time=start,
        end_time=end,
        use_velocity=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    extent = [start, end, pitch_min - 0.5, pitch_max + 0.5]

    im = ax.imshow(
        roll,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    _set_pitch_axis(ax, pitch_min, pitch_max)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Velocity / note intensity")

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=180, facecolor="white")

    return fig


def plot_event_roll_comparison(
    reference_events: Iterable,
    predicted_events: Iterable,
    title: str = "Reference vs predicted MIDI piano rolls",
    reference_label: str = "Reference",
    predicted_label: str = "Prediction",
    save_path: str | Path | None = None,
    *,
    fps: float = DEMO_FPS,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = None,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    figsize=(14, 7),
):
    """Two-panel piano-roll comparison built from decoded note events."""
    ref_parsed = _parse_events(reference_events)
    pred_parsed = _parse_events(predicted_events)
    both = list(ref_parsed) + list(pred_parsed)
    start, end = _resolve_time_window(both, start_time, end_time, window_duration)

    ref_roll = note_events_to_roll(
        ref_parsed,
        fps=fps,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        start_time=start,
        end_time=end,
        use_velocity=True,
    )
    pred_roll = note_events_to_roll(
        pred_parsed,
        fps=fps,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        start_time=start,
        end_time=end,
        use_velocity=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
    extent = [start, end, pitch_min - 0.5, pitch_max + 0.5]

    im0 = axes[0].imshow(
        ref_roll,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title(reference_label)
    _set_pitch_axis(axes[0], pitch_min, pitch_max)

    im1 = axes[1].imshow(
        pred_roll,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(predicted_label)
    axes[1].set_xlabel("Time (s)")
    _set_pitch_axis(axes[1], pitch_min, pitch_max)

    for ax in axes:
        ax.grid(True, axis="x", alpha=0.20)

    fig.colorbar(im0, ax=axes[0], pad=0.01, label="Reference intensity")
    fig.colorbar(im1, ax=axes[1], pad=0.01, label="Predicted velocity / intensity")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=180, facecolor="white")

    return fig


def plot_event_bar_comparison(
    reference_events: Iterable,
    predicted_events: Iterable,
    title: str = "Reference vs predicted MIDI note events",
    reference_label: str = "Reference",
    predicted_label: str = "Prediction",
    save_path: str | Path | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = 60.0,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    figsize=(14, 7),
):
    """Duration-aware event comparison using horizontal MIDI note bars.

    This is better than scatter for dissertation/demo comparison because it shows
    both onset timing and note duration.
    """
    ref_parsed = _parse_events(reference_events)
    pred_parsed = _parse_events(predicted_events)
    both = list(ref_parsed) + list(pred_parsed)
    start, end = _resolve_time_window(both, start_time, end_time, window_duration)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    _draw_note_bars(
        axes[0],
        ref_parsed,
        start_time=start,
        end_time=end,
        color_mode="fixed",
        fixed_color="#4A4A4A",
        alpha=0.75,
        linewidth=0.05,
        edgecolor="#222222",
    )
    axes[0].set_title(reference_label)
    _set_pitch_axis(axes[0], pitch_min, pitch_max)

    sm = _draw_note_bars(
        axes[1],
        pred_parsed,
        start_time=start,
        end_time=end,
        color_mode="velocity",
        cmap_name="viridis",
        alpha=0.88,
        linewidth=0.10,
        edgecolor="black",
        velocity_norm=_velocity_norm(pred_parsed),
    )
    axes[1].set_title(predicted_label)
    axes[1].set_xlabel("Time (s)")
    _set_pitch_axis(axes[1], pitch_min, pitch_max)

    if sm is not None:
        fig.colorbar(sm, ax=axes[1], pad=0.01, label="Predicted velocity")

    for ax in axes:
        ax.set_xlim(start, end)
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=180, facecolor="white")

    return fig


def plot_event_roll_diff(
    reference_events: Iterable,
    predicted_events: Iterable,
    title: str = "Decoded MIDI occupancy difference",
    save_path: str | Path | None = None,
    *,
    fps: float = DEMO_FPS,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = 60.0,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    figsize=(14, 5),
):
    """Frame-occupancy difference built from decoded note events.

    Green = overlap/match, red = predicted extra occupancy, blue = missed reference occupancy.
    This is a qualitative visual aid, not a replacement for mir_eval metrics.
    """
    ref_parsed = _parse_events(reference_events)
    pred_parsed = _parse_events(predicted_events)
    both = list(ref_parsed) + list(pred_parsed)
    start, end = _resolve_time_window(both, start_time, end_time, window_duration)

    ref_roll = note_events_to_roll(
        ref_parsed,
        fps=fps,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        start_time=start,
        end_time=end,
        use_velocity=False,
    ) > 0
    pred_roll = note_events_to_roll(
        pred_parsed,
        fps=fps,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        start_time=start,
        end_time=end,
        use_velocity=False,
    ) > 0

    n_frames = min(ref_roll.shape[1], pred_roll.shape[1])
    ref_roll = ref_roll[:, :n_frames]
    pred_roll = pred_roll[:, :n_frames]

    rgb = np.ones((ref_roll.shape[0], n_frames, 3), dtype=np.float32)

    matched = ref_roll & pred_roll
    extra = pred_roll & (~ref_roll)
    missed = ref_roll & (~pred_roll)

    rgb[matched] = np.array([0.20, 0.70, 0.25], dtype=np.float32)
    rgb[extra] = np.array([0.90, 0.20, 0.20], dtype=np.float32)
    rgb[missed] = np.array([0.20, 0.35, 0.90], dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    extent = [start, end, pitch_min - 0.5, pitch_max + 0.5]

    ax.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    _set_pitch_axis(ax, pitch_min, pitch_max)
    ax.legend(
        handles=[
            Patch(facecolor=(0.20, 0.70, 0.25), label="Overlap"),
            Patch(facecolor=(0.90, 0.20, 0.20), label="Extra prediction"),
            Patch(facecolor=(0.20, 0.35, 0.90), label="Missed reference"),
        ],
        loc="upper right",
    )
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=180, facecolor="white")

    return fig


# Backwards-compatible name used in your existing notebook.
def plot_pred_vs_gt_events(
    pred_events: Iterable[DemoNoteEvent],
    gt_events: Iterable[DemoNoteEvent],
    title: str = "Predicted vs evaluation ground-truth note events",
    max_time: float | None = None,
    save_path: str | Path | None = None,
):
    return plot_event_bar_comparison(
        reference_events=gt_events,
        predicted_events=pred_events,
        title=title,
        reference_label="Evaluation GT MIDI from cached rolls",
        predicted_label="Predicted MIDI",
        save_path=save_path,
        start_time=0.0,
        end_time=max_time,
        window_duration=None if max_time is not None else 60.0,
    )


# ---------------------------------------------------------------------------
# Raw model-posterior diagnostic plots
# ---------------------------------------------------------------------------

def plot_raw_frame_posterior(
    frame_roll,
    title: str = "Raw frame posterior roll before MIDI decoding",
    save_path: str | Path | None = None,
    *,
    fps: float = DEMO_FPS,
    frame_threshold: float | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
    window_duration: float | None = None,
    figsize=(14, 5),
):
    """Plot the raw frame-head model output.

    Diagnostic only: this is not the same as the final decoded MIDI.
    """
    frame_arr = _to_numpy(frame_roll)

    if frame_arr.ndim != 2:
        raise ValueError(f"Expected 2D frame roll, got shape {frame_arr.shape}")

    if frame_arr.shape[1] == 88:
        img = frame_arr.T
    elif frame_arr.shape[0] == 88:
        img = frame_arr
    else:
        raise ValueError(f"Expected one dimension to be 88, got shape {frame_arr.shape}")

    total_duration = img.shape[1] / fps
    start = max(0.0, float(start_time))

    if window_duration is not None:
        end = min(total_duration, start + float(window_duration))
    elif end_time is not None:
        end = min(total_duration, float(end_time))
    else:
        end = total_duration

    if end <= start:
        end = min(total_duration, start + 1.0)

    start_idx = max(0, int(np.floor(start * fps)))
    end_idx = min(img.shape[1], max(start_idx + 1, int(np.ceil(end * fps))))
    img = img[:, start_idx:end_idx]

    if frame_threshold is not None:
        img_to_show = (img >= frame_threshold).astype(np.float32)
        cbar_label = f"Thresholded frame activation at {frame_threshold:.2f}"
    else:
        img_to_show = img.astype(np.float32)
        cbar_label = "Frame activation probability"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    extent = [start, end, PIANO_LOW - 0.5, PIANO_HIGH + 0.5]

    im = ax.imshow(
        img_to_show,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    _set_pitch_axis(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=180, facecolor="white")

    return fig


def plot_piano_roll_side_by_side(
    pred_frame,
    gt_frame=None,
    n_frames: int | None = None,
    title: str = "Raw frame-roll comparison",
    save_path: str | Path | None = None,
    frame_threshold: float = 0.4,
):
    """Legacy diagnostic raw frame-roll plot.

    Kept for compatibility with old notebook cells. For final predicted MIDI display,
    prefer plot_decoded_event_roll() and plot_midi_event_bars().
    """
    pred = _to_numpy(pred_frame)
    gt = _to_numpy(gt_frame) if gt_frame is not None else None

    if n_frames is None:
        n_frames = pred.shape[0]
        if gt is not None:
            n_frames = min(n_frames, gt.shape[0])

    n_frames = int(n_frames)

    pred_img = (pred[:n_frames].T > frame_threshold).astype(np.float32)
    duration_s = n_frames / DEMO_FPS
    extent = [0, duration_s, PIANO_LOW - 0.5, PIANO_HIGH + 0.5]

    if gt is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.imshow(
            pred_img,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        _set_pitch_axis(ax)
    else:
        gt_img = (gt[:n_frames].T > 0.5).astype(np.float32)

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True)

        axes[0].imshow(
            gt_img,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        axes[0].set_title("Cached GT frame label roll")

        axes[1].imshow(
            pred_img,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        axes[1].set_title("Raw predicted frame roll")

        for ax in axes:
            _set_pitch_axis(ax)

        axes[1].set_xlabel("Time (s)")
        fig.suptitle(title)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")

    return fig


def plot_roll_diff(
    pred_frame,
    gt_frame,
    frame_threshold: float = 0.4,
    max_frames: int | None = None,
    title: str = "Raw frame-roll difference: green=overlap, red=extra, blue=missed",
    save_path: str | Path | None = None,
):
    """Legacy raw frame-roll difference plot.

    This compares thresholded frame activations, not decoded note events.
    """
    pred = _to_numpy(pred_frame)
    gt = _to_numpy(gt_frame)

    n = min(pred.shape[0], gt.shape[0])
    if max_frames is not None:
        n = min(n, int(max_frames))

    pred = pred[:n]
    gt = gt[:n]

    pred_bin = pred > frame_threshold
    gt_bin = gt > 0.5

    rgb = np.ones((pred_bin.shape[1], pred_bin.shape[0], 3), dtype=np.float32)

    matched = (pred_bin & gt_bin).T
    extra = (pred_bin & (~gt_bin)).T
    missed = ((~pred_bin) & gt_bin).T

    rgb[matched] = np.array([0.20, 0.70, 0.25], dtype=np.float32)
    rgb[extra] = np.array([0.90, 0.20, 0.20], dtype=np.float32)
    rgb[missed] = np.array([0.20, 0.35, 0.90], dtype=np.float32)

    duration_s = n / DEMO_FPS
    extent = [0, duration_s, PIANO_LOW - 0.5, PIANO_HIGH + 0.5]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.imshow(rgb, aspect="auto", origin="lower", interpolation="nearest", extent=extent)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    _set_pitch_axis(ax)

    ax.legend(
        handles=[
            Patch(facecolor=(0.20, 0.70, 0.25), label="Overlap"),
            Patch(facecolor=(0.90, 0.20, 0.20), label="Extra prediction"),
            Patch(facecolor=(0.20, 0.35, 0.90), label="Missed GT"),
        ],
        loc="upper right",
    )

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")

    return fig


# ---------------------------------------------------------------------------
# Original MIDI / sustain visualisation
# ---------------------------------------------------------------------------

def plot_midi_with_sustain_and_velocity(
    midi_path: str | Path,
    title: str = "Original MAESTRO MIDI: notes, velocity and sustain",
    save_path: str | Path | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = None,
):
    """Plot original MIDI notes as bars plus sustain CC64 below."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = midi_to_events(pm)
    parsed = _parse_events(events)
    start, end = _resolve_time_window(parsed, start_time, end_time, window_duration)

    fig, (ax_notes, ax_pedal) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    sm = _draw_note_bars(
        ax_notes,
        parsed,
        start_time=start,
        end_time=end,
        color_mode="velocity",
        cmap_name="plasma",
        alpha=0.88,
        velocity_norm=_velocity_norm(parsed),
    )

    if sm is not None:
        fig.colorbar(sm, ax=ax_notes, pad=0.01, label="MIDI velocity")

    ax_notes.set_title(title)
    _set_pitch_axis(ax_notes)
    ax_notes.grid(True, axis="x", alpha=0.25)

    cc_points: list[tuple[float, int]] = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64 and start <= float(cc.time) <= end:
                cc_points.append((float(cc.time), int(cc.value)))

    if cc_points:
        cc_points.sort()
        times = [t for t, _ in cc_points]
        values = [v for _, v in cc_points]
        ax_pedal.step(times, values, where="post")
        ax_pedal.axhline(64, linestyle="--", linewidth=1)
        ax_pedal.set_ylim(0, 127)
    else:
        ax_pedal.text(
            0.01,
            0.5,
            "No sustain CC64 events in this window",
            transform=ax_pedal.transAxes,
            va="center",
        )
        ax_pedal.set_ylim(0, 1)

    ax_pedal.set_ylabel("Sustain CC64")
    ax_pedal.set_xlabel("Time (s)")
    ax_pedal.grid(True, axis="x", alpha=0.25)
    ax_pedal.set_xlim(start, end)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=180, bbox_inches="tight", facecolor="white")

    return fig



def render_visual_midi(
    pm_or_path,
    html_path: str | Path | None = None,
    show_inline: bool = False,
    plot_width: int = 1000,
    plot_height: int = 360,
):
    """Render Visual MIDI quietly and safely.

    Visual MIDI is presentation-only. This wrapper suppresses Bokeh/Visual MIDI
    warning spam and prevents optional rendering from breaking transcription.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from visual_midi import Plotter, Preset
    except Exception as exc:
        print(f"Visual MIDI unavailable: could not import visual_midi ({exc})")
        return None

    try:
        pm = pretty_midi.PrettyMIDI(str(pm_or_path)) if isinstance(pm_or_path, (str, Path)) else pm_or_path
        preset = Preset(plot_width=plot_width, plot_height=plot_height)
        plotter = Plotter(preset)
    except Exception as exc:
        print(f"Visual MIDI unavailable: setup failed ({exc})")
        return None

    result = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", message=".*HSL.*")
        warnings.filterwarnings("ignore", message=".*BokehDeprecationWarning.*")

        silent_stdout = io.StringIO()
        silent_stderr = io.StringIO()

        if html_path is not None:
            html_path = Path(html_path)
            html_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with contextlib.redirect_stdout(silent_stdout), contextlib.redirect_stderr(silent_stderr):
                    result = plotter.show(pm, str(html_path))
            except Exception:
                try:
                    with contextlib.redirect_stdout(silent_stdout), contextlib.redirect_stderr(silent_stderr):
                        result = plotter.save(pm, str(html_path))
                except Exception as exc:
                    print(f"Visual MIDI HTML rendering skipped: {exc}")
                    result = None

        if show_inline:
            try:
                with contextlib.redirect_stdout(silent_stdout), contextlib.redirect_stderr(silent_stderr):
                    notebook_result = plotter.show_notebook(pm)
                if notebook_result is not None:
                    result = notebook_result
            except Exception as exc:
                print(f"Visual MIDI inline display skipped: {exc}")

    return result