from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt



from demo.demo_config import HTML_DIR, PLOT_DIR, SAMPLE_RATE, ensure_demo_dirs



def synthesize_pretty_midi(pm, sr: int = SAMPLE_RATE, sf2_path: str | None = None) -> np.ndarray:
    """ FluidSynth + optional piano soundfont."""

    # force piano instrument for demo playback
    for inst in pm.instruments:
        inst.program = 0
        inst.is_drum = False

    try:
        if sf2_path:
            y = pm.fluidsynth(fs=sr, sf2_path=sf2_path)
        else:
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
    save_path=None,
    frame_threshold: float = 0.5,
):


    def _prep(x, threshold=None):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        x = x[:n_frames].T

        # binarise so prediction and GT are visually comparable
        if threshold is not None:
            x = (x > threshold).astype(np.float32)

        return x

    pred_img = _prep(pred_frame, threshold=frame_threshold)
    gt_img = _prep(gt_frame, threshold=0.5)

    if gt_img is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.imshow(
            pred_img,
            aspect="auto",
            origin="lower",
            cmap="gray_r",   # white bg, black notes
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Piano key")
        ax.set_facecolor("white")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

        axes[0].imshow(
            gt_img,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[0].set_title("Ground truth")

        axes[1].imshow(
            pred_img,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[1].set_title("Prediction")

        for ax in axes:
            ax.set_xlabel("Frame")
            ax.set_facecolor("white")

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


def render_bokeh_midi(pm, show_inline: bool = False):
    """Inline-only Bokeh MIDI plot for Colab. No file saving."""
    try:
        from bokeh.io import output_notebook, show
        from visual_midi import Plotter
    except Exception:
        return None

    plotter = Plotter()

    try:
        plot = plotter.show(pm, show=False)
    except TypeError:
        try:
            plot = plotter.show(pm)
        except Exception:
            return None
    except Exception:
        return None

    if plot is not None:
        if hasattr(plot, "width"):
            plot.width = 1000
        if hasattr(plot, "height"):
            plot.height = 320

    if show_inline and plot is not None:
        output_notebook()
        show(plot)

    return plot
