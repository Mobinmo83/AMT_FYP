"""
audio.py — Audio loading and log-mel spectrogram computation.

Design:
  - torchaudio for all I/O and DSP (no librosa dependency).
  - log(mel + 1e-9) formula from jongwook/onsets-and-frames src/mel.py.
  - Module-level singleton _mel_transform to avoid re-building filterbanks.

Papers:
  Hawthorne et al. 2018a §3: N_MELS=229, SR=16000, HOP=512, FMIN=30, FMAX=8000.
  jongwook/onsets-and-frames src/mel.py: log(mel + 1e-9) formula.
"""

from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Tuple

from .constants import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
    MEL_FMIN,
    MEL_FMAX,
    LOG_OFFSET,
    MAX_SEGMENT_FRAMES,
)

# ---------------------------------------------------------------------------
# Module-level singleton mel transform
# Built once; reused across all calls to wav_to_log_mel in this process.
# ---------------------------------------------------------------------------

def _build_mel_transform(device: torch.device) -> T.MelSpectrogram:
    """
    Construct a torchaudio MelSpectrogram transform with parameters from
    Hawthorne 2018a §3 Table 1.

    Args:
        device: Target torch device.

    Returns:
        Configured MelSpectrogram transform on ``device``.
    """
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window_fn=torch.hann_window,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        n_mels=N_MELS,
        power=2.0,          # power spectrogram before mel filterbank
        normalized=False,
    ).to(device)


# Singleton: keyed by device string so CPU and CUDA can coexist.
_mel_transforms: dict[str, T.MelSpectrogram] = {}


def _get_mel_transform(device: torch.device) -> T.MelSpectrogram:
    """Return (or build) the singleton MelSpectrogram for ``device``."""
    key = str(device)
    if key not in _mel_transforms:
        _mel_transforms[key] = _build_mel_transform(device)
    return _mel_transforms[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(
    path: str | Path,
    target_sr: int = SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and resample to ``target_sr``.

    Uses torchaudio.load() followed by T.Resample() when necessary.
    Converts to mono by averaging across channels.

    Args:
        path:      Path to audio file (.wav, .flac, .mp3, …).
        target_sr: Desired sample rate in Hz. Default: 16000 (Hawthorne 2018a §3).
        mono:      If True, collapse to single channel.

    Returns:
        waveform: Tensor of shape (1, N) — always 1 channel after mono conversion.
        sr:       Sample rate of the returned waveform (== target_sr).

    Shape:
        Output waveform: (1, N_samples)
    """
    waveform, sr = torchaudio.load(str(path))

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Mono conversion: mean across channel dimension
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


def wav_to_log_mel(
    waveform: torch.Tensor,
    mel_transform: Optional[T.MelSpectrogram] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a waveform tensor to a log-mel spectrogram.

    Formula: log(mel + 1e-9)
    Source: jongwook/onsets-and-frames src/mel.py line 27.
    Parameters: Hawthorne 2018a §3 Table 1 (N_MELS=229, FMIN=30, FMAX=8000).

    Args:
        waveform:      Tensor of shape (1, N) or (N,).
        mel_transform: Pre-built MelSpectrogram transform.  If None, the
                       module-level singleton is used (preferred).
        device:        Device to run the transform on.  Defaults to waveform device.

    Returns:
        log_mel: Tensor of shape (N_MELS, T_frames) = (229, T).

    Shape:
        Input:  (1, N_samples) or (N_samples,)
        Output: (229, T_frames)
    """
    if device is None:
        device = waveform.device

    # Ensure waveform is on the right device and has shape (1, N)
    waveform = waveform.to(device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if mel_transform is None:
        mel_transform = _get_mel_transform(device)
    else:
        mel_transform = mel_transform.to(device)

    # mel: (1, N_MELS, T) → squeeze batch dim → (N_MELS, T)
    with torch.no_grad():
        mel = mel_transform(waveform)  # (1, 229, T)
    mel = mel.squeeze(0)              # (229, T)

    # Log compression — jongwook src/mel.py
    log_mel = torch.log(mel + LOG_OFFSET)

    return log_mel


def load_audio_as_log_mel(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convenience wrapper: load audio from disk and return its log-mel spectrogram.

    Intended for inference when you need the full-length spectrogram of a file.

    Args:
        path:   Path to audio file.
        device: Target device. Defaults to CPU.

    Returns:
        log_mel: Tensor of shape (229, T_frames).

    Shape:
        Output: (229, T_frames)
    """
    if device is None:
        device = torch.device("cpu")

    waveform, _ = load_audio(path, target_sr=SAMPLE_RATE, mono=True)
    log_mel = wav_to_log_mel(waveform, device=device)
    return log_mel


def audio_samples_to_frames(n_samples: int) -> int:
    """
    Convert a number of audio samples to the corresponding number of
    mel-spectrogram frames using the global HOP_LENGTH.

    Formula: n_frames = n_samples // HOP_LENGTH
    Consistent with torchaudio.transforms.MelSpectrogram frame count.

    Args:
        n_samples: Number of audio samples.

    Returns:
        Number of frames.
    """
    return n_samples // HOP_LENGTH + 1


def frames_to_audio_samples(n_frames: int) -> int:
    """
    Convert a number of mel-spectrogram frames back to audio samples.

    Formula: n_samples = n_frames * HOP_LENGTH

    Args:
        n_frames: Number of mel spectrogram frames.

    Returns:
        Number of audio samples.
    """
    return n_frames * HOP_LENGTH
