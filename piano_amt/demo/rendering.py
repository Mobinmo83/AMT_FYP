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
from matplotlib.ticker import MultipleLocator

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


# Suppress common Visual MIDI / Bokeh warning spam globally for demo notebooks.
try:
    from bokeh.util.warnings import BokehDeprecationWarning

    warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
except Exception:
    pass

warnings.filterwarnings("ignore", message=".*HSL.*")
warnings.filterwarnings("ignore", message=".*BokehDeprecationWarning.*")


def _velocity_to_midi_value(v) -> int:
    v = float(v)
    if v <= 1.0:
        v = v * 127.0
    return int(np.clip(round(v), 1, 127))




# Audio / MIDI synthesis helpers


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


def midi_path_to_events(midi_path: str | Path) -> list[DemoNoteEvent]:
    """Load a MIDI file and convert it into demo note events.

    Use this for final predicted-MIDI visualisation so the displayed notes are
    guaranteed to match the downloadable MIDI file.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return midi_to_events(pm)


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


def _style_midi_axis(
    ax,
    *,
    start_time: float,
    end_time: float,
    major_tick_sec: float = 5.0,
    minor_tick_sec: float = 1.0,
    facecolor: str = "white",
) -> None:
    """Apply a clean dissertation/demo style to MIDI axes."""
    ax.set_facecolor(facecolor)
    ax.set_xlim(start_time, end_time)

    if major_tick_sec is not None and major_tick_sec > 0:
        ax.xaxis.set_major_locator(MultipleLocator(major_tick_sec))
    if minor_tick_sec is not None and minor_tick_sec > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_sec))

    ax.grid(True, which="major", axis="x", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", axis="x", alpha=0.10, linewidth=0.45)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


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
    facecolor: str = "white",
    major_tick_sec: float = 5.0,
    minor_tick_sec: float = 1.0,
    bar_height: float = 0.78,
    edgecolor: str = "black",
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
    fig.patch.set_facecolor(facecolor)

    if not parsed:
        ax.text(0.5, 0.5, "No note events", ha="center", va="center", transform=ax.transAxes)
        _style_midi_axis(
            ax,
            start_time=start,
            end_time=end,
            major_tick_sec=major_tick_sec,
            minor_tick_sec=minor_tick_sec,
            facecolor=facecolor,
        )
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
            bar_height=bar_height,
            edgecolor=edgecolor,
            velocity_norm=_velocity_norm(parsed),
        )

        if sm is not None:
            cbar = fig.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label("Velocity")

        _set_pitch_axis(ax, min_pitch, max_pitch)
        _style_midi_axis(
            ax,
            start_time=start,
            end_time=end,
            major_tick_sec=major_tick_sec,
            minor_tick_sec=minor_tick_sec,
            facecolor=facecolor,
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=200, facecolor=facecolor)

    return fig








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
    window_duration: float | None = None,
    pitch_min: int = PIANO_LOW,
    pitch_max: int = PIANO_HIGH,
    figsize=(16, 9),
    cmap_name: str = "viridis",
):
    """Duration-aware two-panel MIDI comparison with consistent velocity colouring."""
    ref_parsed = _parse_events(reference_events)
    pred_parsed = _parse_events(predicted_events)
    both = list(ref_parsed) + list(pred_parsed)
    start, end = _resolve_time_window(both, start_time, end_time, window_duration)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    shared_norm = _velocity_norm(both)

    sm0 = _draw_note_bars(
        axes[0],
        ref_parsed,
        start_time=start,
        end_time=end,
        color_mode="velocity",
        cmap_name=cmap_name,
        alpha=0.88,
        linewidth=0.10,
        edgecolor="black",
        velocity_norm=shared_norm,
    )

    axes[0].set_title(reference_label)
    _set_pitch_axis(axes[0], pitch_min, pitch_max)
    _style_midi_axis(axes[0], start_time=start, end_time=end, facecolor="white")

    sm1 = _draw_note_bars(
        axes[1],
        pred_parsed,
        start_time=start,
        end_time=end,
        color_mode="velocity",
        cmap_name=cmap_name,
        alpha=0.88,
        linewidth=0.10,
        edgecolor="black",
        velocity_norm=shared_norm,
    )

    axes[1].set_title(predicted_label)
    axes[1].set_xlabel("Time (s)")
    _set_pitch_axis(axes[1], pitch_min, pitch_max)
    _style_midi_axis(axes[1], start_time=start, end_time=end, facecolor="white")

    sm = sm1 if sm1 is not None else sm0
    if sm is not None:
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.015, fraction=0.025)
        cbar.set_label("MIDI velocity")

    fig.suptitle(title)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=200, facecolor="white")

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
    fig.patch.set_facecolor("white")
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
    _style_midi_axis(ax, start_time=start, end_time=end, facecolor="white")
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




# ---------------------------------------------------------------------------
# Original MIDI / sustain visualisation
# ---------------------------------------------------------------------------
def plot_midi_notes_velocity(
    midi_path: str | Path,
    title: str = "MIDI: notes and velocity",
    save_path: str | Path | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    window_duration: float | None = None,
    figsize=(16, 6),
    cmap_name: str = "viridis",
):
    """Plot MIDI notes as velocity-coloured bars without sustain pedal display."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = midi_to_events(pm)
    parsed = _parse_events(events)
    start, end = _resolve_time_window(parsed, start_time, end_time, window_duration)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("white")

    sm = _draw_note_bars(
        ax,
        parsed,
        start_time=start,
        end_time=end,
        color_mode="velocity",
        cmap_name=cmap_name,
        alpha=0.88,
        velocity_norm=_velocity_norm(parsed),
        linewidth=0.12,
        edgecolor="black",
    )

    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.025)
        cbar.set_label("MIDI velocity")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    _set_pitch_axis(ax)
    _style_midi_axis(ax, start_time=start, end_time=end, facecolor="white")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")

    return fig



def render_visual_midi(
    pm_or_path,
    show_inline: bool = True,
    plot_width: int = 1100,
    plot_height: int = 380,
):
    """Render Visual MIDI inline only for Colab/Jupyter.

    This version:
    - does not save HTML
    - does not save PNG
    - does not use IFrame
    - does not call plotter.show(...)
    - only calls plotter.show_notebook(...)
    - avoids duplicate Visual MIDI output
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("ignore", message=".*HSL.*")
            warnings.filterwarnings("ignore", message=".*BokehDeprecationWarning.*")

            try:
                from bokeh.util.warnings import BokehDeprecationWarning
                warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
            except Exception:
                pass

            from bokeh.io import output_notebook
            from visual_midi import Plotter, Preset

    except Exception as exc:
        print(f"Visual MIDI unavailable: could not import visual_midi/Bokeh ({exc})")
        return None

    try:
        output_notebook(hide_banner=True)
    except Exception:
        pass

    try:
        pm = (
            pretty_midi.PrettyMIDI(str(pm_or_path))
            if isinstance(pm_or_path, (str, Path))
            else pm_or_path
        )

        preset = Preset(
            plot_width=int(plot_width),
            plot_height=int(plot_height),
        )

        plotter = Plotter(preset)

    except Exception as exc:
        print(f"Visual MIDI unavailable: setup failed ({exc})")
        return None

    if not show_inline:
        return None

    silent_stdout = io.StringIO()
    silent_stderr = io.StringIO()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", message=".*HSL.*")
        warnings.filterwarnings("ignore", message=".*BokehDeprecationWarning.*")

        try:
            from bokeh.util.warnings import BokehDeprecationWarning
            warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
        except Exception:
            pass

        try:
            # show_notebook already displays the Bokeh MIDI view inline.
            # Do not wrap this function call inside display(...).
            with contextlib.redirect_stdout(silent_stdout), contextlib.redirect_stderr(silent_stderr):
                return plotter.show_notebook(pm)

        except Exception as exc:
            print(f"Visual MIDI inline rendering skipped: {exc}")
            return None
