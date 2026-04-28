from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import soundfile as sf

from demo.demo_config import AUDIO_DIR, DEFAULT_SF2_PATHS, HTML_DIR, SAMPLE_RATE, ensure_demo_dirs
from demo.inference import DemoNoteEvent


PIANO_PROGRAMS = {
    "Acoustic Grand": 0,
    "Bright Acoustic": 1,
    "Electric Grand": 2,
    "Honky-tonk": 3,
    "Electric Piano": 4,
}


def clone_pretty_midi(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mid") as tmp:
        pm.write(tmp.name)
        return pretty_midi.PrettyMIDI(tmp.name)


def apply_piano_program(pm: pretty_midi.PrettyMIDI, piano_sound: str = "Acoustic Grand") -> pretty_midi.PrettyMIDI:
    program = PIANO_PROGRAMS.get(piano_sound, 0)
    pm2 = clone_pretty_midi(pm)
    for inst in pm2.instruments:
        inst.program = program
        inst.is_drum = False
    return pm2


def find_default_sf2() -> str | None:
    for p in DEFAULT_SF2_PATHS:
        if Path(p).exists():
            return p
    return None


def synthesize_pretty_midi(
    pm: pretty_midi.PrettyMIDI,
    sr: int = SAMPLE_RATE,
    piano_sound: str = "Acoustic Grand",
    sf2_path: str | None = None,
) -> np.ndarray:
    """Synthesize MIDI with one of five selectable GM piano programs."""
    pm2 = apply_piano_program(pm, piano_sound)
    sf2_path = sf2_path or find_default_sf2()
    try:
        if sf2_path:
            y = pm2.fluidsynth(fs=sr, sf2_path=sf2_path)
        else:
            y = pm2.fluidsynth(fs=sr)
    except Exception:
        # Fallback keeps the notebook usable without FluidSynth.
        y = pm2.synthesize(fs=sr)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        y = np.zeros(sr, dtype=np.float32)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs > 0:
        y = y / max_abs
    return y


def save_audio_wav(y: np.ndarray, output_path: str | Path, sr: int = SAMPLE_RATE) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, sr)
    return output_path


def synthesize_and_save(
    pm: pretty_midi.PrettyMIDI,
    output_path: str | Path,
    piano_sound: str = "Acoustic Grand",
    sr: int = SAMPLE_RATE,
) -> Path:
    y = synthesize_pretty_midi(pm, sr=sr, piano_sound=piano_sound)
    return save_audio_wav(y, output_path, sr=sr)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_piano_roll_side_by_side(
    pred_frame,
    gt_frame=None,
    n_frames: int = 900,
    title: str = "Piano-roll comparison",
    save_path: str | Path | None = None,
    frame_threshold: float = 0.4,
):
    def _prep(x, threshold=None):
        if x is None:
            return None
        x = _to_numpy(x)[:n_frames].T
        if threshold is not None:
            x = (x > threshold).astype(np.float32)
        return x

    pred_img = _prep(pred_frame, threshold=frame_threshold)
    gt_img = _prep(gt_frame, threshold=0.5)
    if gt_img is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(pred_img, aspect="auto", origin="lower", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Piano key")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
        axes[0].imshow(gt_img, aspect="auto", origin="lower", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title("Ground truth cached label roll")
        axes[1].imshow(pred_img, aspect="auto", origin="lower", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title("Prediction roll")
        for ax in axes:
            ax.set_xlabel("Frame")
        axes[0].set_ylabel("Piano key")
        fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight", facecolor="white")
    return fig


def plot_roll_diff(
    pred_frame,
    gt_frame,
    frame_threshold: float = 0.4,
    max_frames: int = 900,
    title: str = "Diff roll: green=match, red=extra, blue=missed",
    save_path: str | Path | None = None,
):
    pred = _to_numpy(pred_frame)[:max_frames]
    gt = _to_numpy(gt_frame)[:max_frames]
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]
    pred_bin = pred > frame_threshold
    gt_bin = gt > 0.5
    rgb = np.ones((pred_bin.shape[1], pred_bin.shape[0], 3), dtype=np.float32)
    matched = (pred_bin & gt_bin).T
    extra = (pred_bin & (~gt_bin)).T
    missed = ((~pred_bin) & gt_bin).T
    rgb[matched] = np.array([0.20, 0.75, 0.25], dtype=np.float32)
    rgb[extra] = np.array([0.95, 0.20, 0.20], dtype=np.float32)
    rgb[missed] = np.array([0.20, 0.40, 0.95], dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.imshow(rgb, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Piano key")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
    return fig


def midi_to_events(pm: pretty_midi.PrettyMIDI) -> list[DemoNoteEvent]:
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if 21 <= n.pitch <= 108 and n.end > n.start:
                events.append(DemoNoteEvent(n.pitch, n.start, n.end, n.velocity))
    events.sort(key=lambda e: (e.onset_sec, e.pitch, e.offset_sec))
    return events


def plot_note_events_colored(
    events: Iterable[DemoNoteEvent],
    title: str = "Decoded note events",
    save_path: str | Path | None = None,
):
    events = list(events)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    if events:
        starts = np.array([e.onset_sec for e in events])
        durations = np.array([max(e.offset_sec - e.onset_sec, 1e-3) for e in events])
        pitches = np.array([e.pitch for e in events])
        velocities = np.array([e.velocity for e in events])
        coll = ax.scatter(starts, pitches, c=velocities, s=np.clip(durations * 60, 8, 220), cmap="viridis", alpha=0.85)
        cbar = fig.colorbar(coll, ax=ax)
        cbar.set_label("Velocity")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_ylim(20, 109)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
    return fig


def plot_pred_vs_gt_events(
    pred_events: Iterable[DemoNoteEvent],
    gt_events: Iterable[DemoNoteEvent],
    title: str = "Predicted vs evaluation ground truth note events",
    max_time: float | None = None,
    save_path: str | Path | None = None,
):
    pred_events = list(pred_events)
    gt_events = list(gt_events)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    if gt_events:
        gt_starts = np.array([e.onset_sec for e in gt_events])
        gt_pitches = np.array([e.pitch for e in gt_events])
        if max_time is not None:
            mask = gt_starts <= max_time
            gt_starts, gt_pitches = gt_starts[mask], gt_pitches[mask]
        ax.scatter(gt_starts, gt_pitches, marker="o", s=20, alpha=0.40, label="GT eval-roll MIDI")
    if pred_events:
        pred_starts = np.array([e.onset_sec for e in pred_events])
        pred_pitches = np.array([e.pitch for e in pred_events])
        pred_vel = np.array([e.velocity for e in pred_events])
        if max_time is not None:
            mask = pred_starts <= max_time
            pred_starts, pred_pitches, pred_vel = pred_starts[mask], pred_pitches[mask], pred_vel[mask]
        coll = ax.scatter(pred_starts, pred_pitches, marker="x", c=pred_vel, s=35, alpha=0.85, label="Prediction")
        cbar = fig.colorbar(coll, ax=ax)
        cbar.set_label("Predicted velocity")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_ylim(20, 109)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
    return fig


def plot_midi_with_sustain_and_velocity(
    midi_path: str | Path,
    title: str = "Original MAESTRO MIDI: velocity and sustain",
    save_path: str | Path | None = None,
):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = midi_to_events(pm)
    fig, (ax_notes, ax_pedal) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    if events:
        starts = np.array([e.onset_sec for e in events])
        durations = np.array([max(e.offset_sec - e.onset_sec, 1e-3) for e in events])
        pitches = np.array([e.pitch for e in events])
        velocities = np.array([e.velocity for e in events])
        coll = ax_notes.scatter(starts, pitches, c=velocities, s=np.clip(durations * 60, 8, 220), cmap="plasma", alpha=0.85)
        cbar = fig.colorbar(coll, ax=ax_notes)
        cbar.set_label("Velocity")
    ax_notes.set_ylabel("MIDI pitch")
    ax_notes.set_ylim(20, 109)
    ax_notes.set_title(title)

    cc_points = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64:
                cc_points.append((cc.time, cc.value))
    if cc_points:
        cc_points.sort()
        times = [t for t, _ in cc_points]
        values = [v for _, v in cc_points]
        ax_pedal.step(times, values, where="post")
        ax_pedal.axhline(64, linestyle="--", linewidth=1)
        ax_pedal.set_ylim(0, 127)
    else:
        ax_pedal.text(0.01, 0.5, "No sustain CC64 events found", transform=ax_pedal.transAxes, va="center")
        ax_pedal.set_ylim(0, 1)
    ax_pedal.set_ylabel("Sustain")
    ax_pedal.set_xlabel("Time (s)")
    ax_notes.grid(True, alpha=0.25)
    ax_pedal.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
    return fig


def render_visual_midi(pm_or_path, html_path: str | Path | None = None, show_inline: bool = False):
    """Render Visual MIDI safely in notebooks and optionally save HTML."""
    try:
        from visual_midi import Plotter, Preset
    except Exception:
        return None

    pm = pretty_midi.PrettyMIDI(str(pm_or_path)) if isinstance(pm_or_path, (str, Path)) else pm_or_path
    preset = Preset(plot_width=1000, plot_height=360)
    plotter = Plotter(preset)
    result = None
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = plotter.show(pm, str(html_path))
        except TypeError:
            try:
                result = plotter.save(pm, str(html_path))
            except Exception:
                result = None
    if show_inline:
        try:
            result = plotter.show_notebook(pm)
        except Exception:
            pass
    return result
