"""
constants.py — Global hyperparameters for the piano AMT pipeline.

Every value cites the paper/section it originates from.
"""

# ---------------------------------------------------------------------------
# Audio parameters
# Paper: Hawthorne et al. 2018a "Onsets and Frames" §3 (Table 1)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
"""16 kHz sample rate — Hawthorne 2018a §3, jongwook/onsets-and-frames src/constants.py"""

N_FFT: int = 2048
"""FFT window size — jongwook/onsets-and-frames src/constants.py"""

HOP_LENGTH: int = 512
"""Hop size in samples — Hawthorne 2018a §3: yields 31.25 frames/sec at 16 kHz"""

WIN_LENGTH: int = 2048
"""Window length — jongwook/onsets-and-frames src/constants.py"""

# ---------------------------------------------------------------------------
# Mel spectrogram parameters
# Paper: Hawthorne et al. 2018a §3 / jongwook/onsets-and-frames src/constants.py
# ---------------------------------------------------------------------------

N_MELS: int = 229
"""Number of mel filterbank bins — Hawthorne 2018a §3 Table 1"""

MEL_FMIN: float = 30.0
"""Lowest mel frequency (Hz) — Hawthorne 2018a §3 Table 1"""

MEL_FMAX: float = 8000.0
"""Highest mel frequency (Hz) — Hawthorne 2018a §3 Table 1"""

LOG_OFFSET: float = 1e-9
"""Numerical stabiliser inside log: log(mel + 1e-9) — jongwook src/mel.py line 27"""

# ---------------------------------------------------------------------------
# Frame rate (derived constant)
# ---------------------------------------------------------------------------

FRAMES_PER_SECOND: float = SAMPLE_RATE / HOP_LENGTH  # = 31.25
"""31.25 frames/sec — Hawthorne 2018a §3: SAMPLE_RATE / HOP_LENGTH"""

# ---------------------------------------------------------------------------
# Piano key range
# Paper: Hawthorne 2018a §3; standard 88-key piano MIDI range
# ---------------------------------------------------------------------------

MIN_MIDI: int = 21
"""Lowest piano MIDI note — A0, standard 88-key piano (Hawthorne 2018a §3)"""

MAX_MIDI: int = 108
"""Highest piano MIDI note — C8, standard 88-key piano (Hawthorne 2018a §3)"""

N_KEYS: int = MAX_MIDI - MIN_MIDI + 1  # = 88
"""Total piano keys — 88, derived from MIN/MAX_MIDI (Hawthorne 2018a §3)"""

# ---------------------------------------------------------------------------
# Label encoding parameters
# Paper: Hawthorne 2018a §3.1 "Label Design"
# ---------------------------------------------------------------------------

ONSET_WINDOW_FRAMES: int = 1
"""Number of frames to mark as onset around the true onset frame.
Hawthorne 2018a §3.1: onset label spans 1 frame either side (width=1 in practice).
Using 1 here to stay consistent with jongwook reference implementation."""

OFFSET_WINDOW_FRAMES: int = 1
"""Number of frames to mark around the true offset frame.
Included to match jongwook/onsets-and-frames offset head improvement."""

VELOCITY_SCALE: float = 128.0
"""Normalise raw MIDI velocity [0,127] → [0,1] by dividing by 128.
Hawthorne 2018a §3.1: velocity target is v/128."""

# ---------------------------------------------------------------------------
# Segment (crop) parameters
# Paper: jongwook/onsets-and-frames src/dataset.py
# ---------------------------------------------------------------------------

MAX_SEGMENT_FRAMES: int = 640
"""Random crop length in frames — jongwook src/dataset.py line 41: 327 680 / 512 = 640"""

MAX_SEGMENT_SAMPLES: int = MAX_SEGMENT_FRAMES * HOP_LENGTH  # = 327 680
"""Random crop length in audio samples — jongwook src/dataset.py: ~20 s at 16 kHz"""

# ---------------------------------------------------------------------------
# MAESTRO CSV column names
# Paper: Hawthorne et al. 2018b "MAESTRO Dataset" §3 / README
# ---------------------------------------------------------------------------

MAESTRO_AUDIO_COL: str = "audio_filename"
"""Column in maestro.csv holding relative audio file path (MAESTRO v3 schema)"""

MAESTRO_MIDI_COL: str = "midi_filename"
"""Column in maestro.csv holding relative MIDI file path (MAESTRO v3 schema)"""

MAESTRO_SPLIT_COL: str = "split"
"""Column in maestro.csv with values 'train' / 'validation' / 'test'
(Hawthorne 2018b §3: standard splits for AMT benchmarking)"""

MAESTRO_DURATION_COL: str = "duration"
"""Column in maestro.csv holding piece duration in seconds (MAESTRO v3 schema)"""
