"""
evaluate/plots.py — Visualisation utilities for training and evaluation.

Functions:
  plot_training_curves()        — loss curves from metrics.json (report quality)
  plot_piano_roll_comparison()  — GT vs predicted piano roll side by side
  plot_metrics_bar()            — bar chart of F1 scores across heads
  plot_all_run_curves()         — overlay multiple runs on one figure

All functions accept a save_path argument and either save to file (headless
Colab / Drive) or show interactively if save_path is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Matplotlib guard
# ---------------------------------------------------------------------------

def _mpl():
    """Import matplotlib with Agg backend for headless environments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# 1. Training curves from metrics.json
# ---------------------------------------------------------------------------

def plot_training_curves(
    metrics_path: Union[str, Path],
    save_path:    Optional[Union[str, Path]] = None,
    title:        str = "Training curves",
) -> None:
    """
    Plot total loss + per-head losses + learning rate from a run's metrics.json.

    Args:
        metrics_path: Path to metrics.json produced by RunDirectory.log_epoch().
        save_path:    If given, save PNG here; else show interactively.
        title:        Figure suptitle (use the run name for the report).
    """
    plt = _mpl()

    with open(metrics_path) as f:
        h = json.load(f)

    ep = h["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=14)

    # Total loss
    axes[0, 0].plot(ep, h["train_loss"], label="train", linewidth=1.5)
    axes[0, 0].plot(ep, h["val_loss"],   label="val",   linewidth=1.5)
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Per-head losses
    for ax, (t_key, v_key, title_str) in zip(
        [axes[0,1], axes[0,2], axes[1,0], axes[1,1]],
        [
            ("train_onset",  "val_onset",  "Onset BCE"),
            ("train_frame",  "val_frame",  "Frame BCE"),
            ("train_offset", "val_offset", "Offset BCE"),
            ("train_vel",    "val_vel",    "Velocity MSE"),
        ]
    ):
        ax.plot(ep, h.get(t_key, []), label="train", linewidth=1.5)
        ax.plot(ep, h.get(v_key, []), label="val",   linewidth=1.5)
        ax.set_title(title_str)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Learning rate
    axes[1, 2].semilogy(ep, h["lr"], color="green", linewidth=1.5)
    axes[1, 2].set_title("Learning rate (log scale)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Piano roll comparison (GT vs prediction)
# ---------------------------------------------------------------------------

def plot_piano_roll_comparison(
    pred_frame:  Union[torch.Tensor, np.ndarray],  # (T, 88)
    gt_frame:    Union[torch.Tensor, np.ndarray],  # (T, 88)
    pred_onset:  Optional[Union[torch.Tensor, np.ndarray]] = None,
    gt_onset:    Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_frames:    int = 400,
    title:       str = "Piano roll: GT vs prediction",
    save_path:   Optional[Union[str, Path]] = None,
) -> None:
    """
    Side-by-side piano-roll comparison (ground truth left, prediction right).
    Useful for visual quality assessment in the report.

    Args:
        pred_frame: (T, 88) predicted frame roll.
        gt_frame:   (T, 88) ground-truth frame roll.
        pred_onset: (T, 88) predicted onset roll (optional overlay).
        gt_onset:   (T, 88) ground-truth onset roll (optional overlay).
        n_frames:   Number of frames to show (default 400 ≈ 13 s).
        title:      Figure title.
        save_path:  Save PNG here if given.
    """
    plt = _mpl()

    def _np(x):
        if x is None:
            return None
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        return np.asarray(x)

    pf = _np(pred_frame)[:n_frames].T   # (88, n_frames)
    gf = _np(gt_frame)[:n_frames].T

    n_cols = 2
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 5), sharey=True)
    fig.suptitle(title, fontsize=12)

    key_ticks  = [0, 21, 43, 65, 87]
    key_labels = ["A0", "C2", "C4", "C6", "C8"]

    for ax, roll, subtit, onset in zip(
        axes,
        [gf, pf],
        ["Ground truth", "Prediction"],
        [_np(gt_onset), _np(pred_onset)],
    ):
        ax.imshow(roll, aspect="auto", origin="lower", cmap="Greens",
                  vmin=0, vmax=1)
        if onset is not None:
            on = onset[:n_frames].T
            ax.imshow(
                np.ma.masked_where(on < 0.5, on),
                aspect="auto", origin="lower", cmap="Reds", alpha=0.7,
                vmin=0, vmax=1,
            )
        ax.set_title(subtit)
        ax.set_xlabel("Frame")
        ax.set_yticks(key_ticks)
        ax.set_yticklabels(key_labels)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_bar(
    metrics:   Dict[str, float],
    run_name:  str = "",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Horizontal bar chart of the main F1 / accuracy metrics for one run.

    Args:
        metrics:   Dict from compute_metrics() or summary_metrics.json.
        run_name:  Label for the figure title.
        save_path: Save PNG here if given.
    """
    plt = _mpl()

    keys = [
        ("frame_f1",                "Frame F1"),
        ("onset_f1",                "Onset F1"),
        ("note_with_offset_f1",     "Note + offset F1"),
        ("note_with_offset_vel_f1", "Note + offset + vel F1"),
        ("frame_accuracy",          "Frame accuracy"),
    ]
    labels = []
    values = []
    for k, label in keys:
        if k in metrics:
            labels.append(label)
            values.append(metrics[k])

    if not labels:
        print("No plottable metrics found.")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#2196F3", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(f"Evaluation metrics — {run_name}")
    ax.grid(True, axis="x", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Metrics bar chart saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Overlay multiple runs' training curves
# ---------------------------------------------------------------------------

def plot_all_run_curves(
    runs_dir:  Union[str, Path],
    metric:    str = "val_loss",
    save_path: Optional[Union[str, Path]] = None,
    title:     str = "All runs — training comparison",
) -> None:
    """
    Load metrics.json from every run in runs_dir and overlay them on one plot.

    Useful for the dissertation results section comparing runs.

    Args:
        runs_dir:  Directory containing run subdirectories (each with metrics.json).
        metric:    Which metric to plot: "val_loss", "train_loss", etc.
        save_path: Save PNG here if given.
        title:     Figure title.
    """
    plt = _mpl()

    runs_dir = Path(runs_dir)
    metrics_files = sorted(runs_dir.glob("*/metrics.json"))

    if not metrics_files:
        print(f"No metrics.json files found under {runs_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=13)

    for mf in metrics_files:
        run_name = mf.parent.name
        with open(mf) as f:
            h = json.load(f)
        if metric in h and h["epoch"]:
            ax.plot(h["epoch"], h[metric], label=run_name, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"All-run comparison saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
