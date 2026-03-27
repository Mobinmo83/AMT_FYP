"""
utils/viz.py — Visualisation utilities for piano AMT pipeline inspection.

All functions:
  - Guard against matplotlib being unavailable (_check_mpl).
  - Accept both torch.Tensor and np.ndarray inputs.
  - Use imshow with origin="lower" so frequency increases upward.

THE alignment check is plot_mel_with_labels():
  Bright bands in the mel (top) must line up with active rows in the frame
  roll (bottom). Any misalignment indicates a pipeline bug (fps, HOP_LENGTH,
  or time-shift error).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Guard: matplotlib is optional (headless environments, unit tests)
# ---------------------------------------------------------------------------

def _check_mpl() -> None:
    """Raise ImportError with a helpful message if matplotlib is absent."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        ) from exc


def _to_numpy(x: Union["torch.Tensor", np.ndarray]) -> np.ndarray:
    """Convert Tensor or ndarray to a float32 numpy array (CPU)."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


# Piano key labels for Y-axis ticks on roll plots
_KEY_TICKS  = [0, 21, 43, 65, 87]           # key indices (0-based)
_KEY_LABELS = ["A0", "C2", "C4", "C6", "C8"]  # corresponding note names


# ---------------------------------------------------------------------------
# plot_mel
# ---------------------------------------------------------------------------

def plot_mel(
    mel:     Union["torch.Tensor", np.ndarray],
    title:   str = "Log-Mel Spectrogram",
    figsize: Tuple[int, int] = (14, 4),
    vmin:    float = -10.0,
    vmax:    float = 2.0,
) -> None:
    """
    Display a log-mel spectrogram as a colour image.

    Args:
        mel:    Array of shape (229, T_frames) — log-mel spectrogram.
        title:  Figure title.
        figsize: Matplotlib figure size in inches.
        vmin:   Colour scale minimum (log units; default -10.0).
        vmax:   Colour scale maximum (log units; default 2.0).

    Shape:
        mel: (N_MELS, T_frames) = (229, T)
    """
    _check_mpl()
    import matplotlib.pyplot as plt

    mel_np = _to_numpy(mel)  # (229, T)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mel_np,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    plt.colorbar(im, ax=ax, label="Log amplitude")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# plot_piano_roll
# ---------------------------------------------------------------------------

def plot_piano_roll(
    onset:    Union["torch.Tensor", np.ndarray],
    frame:    Union["torch.Tensor", np.ndarray],
    offset:   Optional[Union["torch.Tensor", np.ndarray]] = None,
    velocity: Optional[Union["torch.Tensor", np.ndarray]] = None,
    title:    str = "Piano Roll Labels",
    figsize:  Tuple[int, int] = (14, 8),
) -> None:
    """
    Display piano-roll label tensors as stacked imshow subplots.

    Always shows onset and frame; optionally shows offset and velocity.
    Y-axis ticks are labelled with piano note names.

    Args:
        onset:    Array (T, 88) — onset piano roll.
        frame:    Array (T, 88) — frame piano roll.
        offset:   Array (T, 88) — offset piano roll (optional).
        velocity: Array (T, 88) — velocity piano roll (optional).
        title:    Overall figure suptitle.
        figsize:  Matplotlib figure size.

    Shape:
        All inputs: (T_frames, N_KEYS) = (T, 88)
    """
    _check_mpl()
    import matplotlib.pyplot as plt

    panels = [("Onset",    _to_numpy(onset).T)]
    panels.append(("Frame",   _to_numpy(frame).T))
    if offset is not None:
        panels.append(("Offset",  _to_numpy(offset).T))
    if velocity is not None:
        panels.append(("Velocity", _to_numpy(velocity).T))

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (label, roll) in zip(axes, panels):
        ax.imshow(
            roll,
            aspect="auto",
            origin="lower",
            cmap="inferno",
        )
        ax.set_ylabel(label)
        ax.set_yticks(_KEY_TICKS)
        ax.set_yticklabels(_KEY_LABELS)

    axes[-1].set_xlabel("Frame")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# plot_mel_with_labels — THE alignment check
# ---------------------------------------------------------------------------

def plot_mel_with_labels(
    mel:      Union["torch.Tensor", np.ndarray],
    frame:    Union["torch.Tensor", np.ndarray],
    onset:    Optional[Union["torch.Tensor", np.ndarray]] = None,
    n_frames: int = 320,
    figsize:  Tuple[int, int] = (14, 6),
    title:    str = "Mel vs Frame Roll — Alignment Check",
) -> None:
    """
    Plot mel spectrogram (top) and frame piano roll (bottom) with a shared
    x-axis for the alignment check.

    If audio-label alignment is correct, bright horizontal bands in the mel
    (harmonics of active notes) will line up vertically with bright columns
    in the frame roll (active keys). Any misalignment indicates a bug in fps,
    HOP_LENGTH, or time-shift code.

    Args:
        mel:      Array (229, T) — log-mel spectrogram.
        frame:    Array (T, 88) — frame piano roll.
        onset:    Array (T, 88) — onset piano roll (optional overlay).
        n_frames: Number of frames to display (default 320 ≈ 10 s).
        figsize:  Matplotlib figure size.
        title:    Figure suptitle.

    Shape:
        mel:   (229, T)
        frame: (T, 88)
    """
    _check_mpl()
    import matplotlib.pyplot as plt

    mel_np   = _to_numpy(mel)[:, :n_frames]    # (229, n_frames)
    frame_np = _to_numpy(frame)[:n_frames, :].T # (88, n_frames)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Top: mel ---
    axes[0].imshow(
        mel_np,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=-10.0,
        vmax=2.0,
    )
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title("Log-Mel Spectrogram")

    # --- Bottom: frame roll ---
    axes[1].imshow(
        frame_np,
        aspect="auto",
        origin="lower",
        cmap="Greens",
    )
    # Overlay onset in red if provided
    if onset is not None:
        onset_np = _to_numpy(onset)[:n_frames, :].T  # (88, n_frames)
        axes[1].imshow(
            np.ma.masked_where(onset_np < 0.5, onset_np),
            aspect="auto",
            origin="lower",
            cmap="Reds",
            alpha=0.7,
        )
    axes[1].set_yticks(_KEY_TICKS)
    axes[1].set_yticklabels(_KEY_LABELS)
    axes[1].set_ylabel("Piano key")
    axes[1].set_xlabel("Frame")
    axes[1].set_title("Frame Roll (+ onset overlay)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# plot_batch_sample
# ---------------------------------------------------------------------------

def plot_batch_sample(
    batch: dict,
    idx:   int = 0,
    n_frames: int = 320,
) -> None:
    """
    Inspect and visualise a single sample from a DataLoader batch.

    Prints:
      - audio_path
      - mel.shape and onset.shape
      - Active note counts (sum of onset/frame cols)

    Then calls plot_mel_with_labels() and plot_piano_roll() for a full
    visual inspection of the sample.

    Args:
        batch:    Dict returned by piano_amt_collate().
        idx:      Index of the sample within the batch (default 0).
        n_frames: Number of frames to show in alignment check.

    Shape (batch):
        mel:    (B, 229, T)
        onset:  (B, T, 88)
        frame:  (B, T, 88)
        offset: (B, T, 88)
        velocity: (B, T, 88)
    """
    _check_mpl()

    mel      = batch["mel"][idx]        # (229, T)
    onset    = batch["onset"][idx]      # (T, 88)
    frame    = batch["frame"][idx]      # (T, 88)
    offset   = batch.get("offset")
    velocity = batch.get("velocity")

    audio_path = (
        batch["audio_path"][idx]
        if "audio_path" in batch else "<unknown>"
    )

    print(f"audio_path : {audio_path}")
    print(f"mel.shape  : {tuple(mel.shape)}")
    print(f"onset.shape: {tuple(onset.shape)}")
    print(f"Active onset frames : {int(onset.sum().item())}")
    print(f"Active frame events : {int(frame.sum().item())}")
    if offset is not None:
        print(f"Active offset events: {int(offset[idx].sum().item())}")

    # Alignment check
    plot_mel_with_labels(
        mel=mel,
        frame=frame,
        onset=onset,
        n_frames=n_frames,
        title=f"Alignment Check — {audio_path.split('/')[-1]}",
    )

    # Full 4-head roll view
    plot_piano_roll(
        onset=onset,
        frame=frame,
        offset=offset[idx] if offset is not None else None,
        velocity=velocity[idx] if velocity is not None else None,
        title="All 4 Label Heads",
    )
