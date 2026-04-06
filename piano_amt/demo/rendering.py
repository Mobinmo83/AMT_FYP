from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from demo.demo_config import HTML_DIR, PLOT_DIR, SAMPLE_RATE, ensure_demo_dirs


def synthesize_pretty_midi(pm, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render PrettyMIDI to audio using fluidsynth when available, else fallback."""
    try:
        y = pm.fluidsynth(fs=sr)
    except Exception:
        y = pm.synthesize(fs=sr)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        y = np.zeros(sr, dtype=np.float32)
    max_abs = np.max(np.abs(y)) if y.size else 0.0
    if max_abs > 0:
        y = y / max_abs
    return y


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
    pred_onset=None,
    gt_onset=None,
    n_frames: int = 800,
    title: str = "Piano roll comparison",
    save_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    pred_frame = _to_numpy(pred_frame)[:n_frames].T
    gt_frame_np = _to_numpy(gt_frame)
    if gt_frame_np is not None:
        gt_frame_np = gt_frame_np[:n_frames].T

    if gt_frame_np is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(pred_frame, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Piano key")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
        axes[0].imshow(gt_frame_np, aspect="auto", origin="lower", cmap="Greens", vmin=0, vmax=1)
        axes[0].set_title("Ground truth")
        axes[1].imshow(pred_frame, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("Prediction")
        for ax in axes:
            ax.set_xlabel("Frame")
        axes[0].set_ylabel("Piano key")
        fig.suptitle(title)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
    return fig


def plot_roll_diff(
    pred_frame,
    gt_frame,
    frame_threshold: float = 0.5,
    max_frames: int = 800,
    title: str = "Diff roll (green=match, red=extra, blue=missed)",
    save_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    pred = _to_numpy(pred_frame)[:max_frames]
    gt = _to_numpy(gt_frame)[:max_frames]
    pred_bin = pred > frame_threshold
    gt_bin = gt > 0.5

    rgb = np.zeros((pred_bin.shape[1], pred_bin.shape[0], 3), dtype=np.float32)
    matched = (pred_bin & gt_bin).T
    extra = (pred_bin & (~gt_bin)).T
    missed = ((~pred_bin) & gt_bin).T

    rgb[matched] = np.array([0.20, 0.80, 0.25], dtype=np.float32)
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


def render_bokeh_midi(pm, html_path: str | Path | None = None, show_inline: bool = False):
    """Create a Bokeh MIDI plot when visual_midi is installed.

    Returns the Bokeh object when successful; otherwise returns ``None``.
    """
    try:
        from bokeh.io import output_file, output_notebook, save, show
        from visual_midi import Preset, Plotter
    except Exception:
        return None

    plotter = Plotter()
    plot = plotter.show(pm, preset=Preset(plot_width=1000, plot_height=320), show=False)

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        output_file(str(html_path))
        save(plot)

    if show_inline:
        output_notebook()
        show(plot)

    return plot
