"""
constants.py — Global hyperparameters for the piano AMT pipeline.

Centralises all fixed values used across audio preprocessing, label encoding,
model input/output shapes, training crops, decoding, evaluation, and MAESTRO
metadata handling.

Design:
  - Audio and mel-spectrogram parameters are defined once and reused by
    audio.py, dataset.py, dataloader.py, model.py, decoding, and evaluation.
  - Frame-rate constants are derived from SAMPLE_RATE and HOP_LENGTH to keep
    timing conversions consistent across the pipeline.
  - Piano key range constants define the standard 88-key MIDI range used by
    all label rolls and model outputs.
  - Label encoding constants control onset, offset, frame, and velocity target
    construction.
  - Segment constants define the fixed training crop length used for random
    training examples.
  - MAESTRO CSV column names are stored here so dataset loading code does not
    duplicate string literals.

Purpose:
  Keeping these values in one file reduces configuration drift and makes the
  preprocessing, training, decoding, and evaluation stages easier to audit.
"""

# ---------------------------------------------------------------------------
# Audio parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000


N_FFT: int = 2048


HOP_LENGTH: int = 512


WIN_LENGTH: int = 2048
"""Window length — jongwook/onsets-and-frames src/constants.py"""

# ---------------------------------------------------------------------------
# Mel spectrogram parameters
# ---------------------------------------------------------------------------

N_MELS: int = 229


MEL_FMIN: float = 30.0


MEL_FMAX: float = 8000.0


LOG_OFFSET: float = 1e-9


# ---------------------------------------------------------------------------
# Frame rate (derived constant)
# ---------------------------------------------------------------------------

FRAMES_PER_SECOND: float = SAMPLE_RATE / HOP_LENGTH  # = 31.25


# ---------------------------------------------------------------------------
# Piano key range
# Paper: Hawthorne 2018a §3; standard 88-key piano MIDI range
# ---------------------------------------------------------------------------

MIN_MIDI: int = 21


MAX_MIDI: int = 108


N_KEYS: int = MAX_MIDI - MIN_MIDI + 1  # = 88


# ---------------------------------------------------------------------------
# Label encoding parameters
# ---------------------------------------------------------------------------

ONSET_WINDOW_FRAMES: int = 1


OFFSET_WINDOW_FRAMES: int = 1


VELOCITY_SCALE: float = 128.0


# ---------------------------------------------------------------------------
# Segment (crop) parameters
# ---------------------------------------------------------------------------

MAX_SEGMENT_FRAMES: int = 640


MAX_SEGMENT_SAMPLES: int = MAX_SEGMENT_FRAMES * HOP_LENGTH  # = 327 680


# ---------------------------------------------------------------------------
# MAESTRO CSV column names
# ---------------------------------------------------------------------------

MAESTRO_AUDIO_COL: str = "audio_filename"


MAESTRO_MIDI_COL: str = "midi_filename"


MAESTRO_SPLIT_COL: str = "split"


MAESTRO_DURATION_COL: str = "duration"

