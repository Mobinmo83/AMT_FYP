from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hugging Face settings
# ---------------------------------------------------------------------------
# Replace these with your public model repo details before sharing the notebook.
HF_REPO_ID = os.getenv("AMT_DEMO_HF_REPO_ID", "YOUR_USERNAME/piano-amt-demo")
HF_CHECKPOINT_FILENAME = os.getenv(
    "AMT_DEMO_HF_CHECKPOINT_FILENAME",
    "checkpoints/best.pt",
)
HF_REPO_TYPE = "model"

# ---------------------------------------------------------------------------
# Model + audio settings
# ---------------------------------------------------------------------------
MODEL_COMPLEXITY = int(os.getenv("AMT_DEMO_MODEL_COMPLEXITY", "48"))
SAMPLE_RATE = int(os.getenv("AMT_DEMO_SAMPLE_RATE", "16000"))
DEFAULT_ONSET_THRESHOLD = float(os.getenv("AMT_DEMO_ONSET_THRESHOLD", "0.50"))
DEFAULT_FRAME_THRESHOLD = float(os.getenv("AMT_DEMO_FRAME_THRESHOLD", "0.50"))
MIN_THRESHOLD = 0.30
MAX_THRESHOLD = 0.90
THRESHOLD_STEP = 0.01

# ---------------------------------------------------------------------------
# Working directories inside Colab / Jupyter
# ---------------------------------------------------------------------------
WORK_DIR = Path(os.getenv("AMT_DEMO_WORK_DIR", "/content/amt_demo"))
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = WORK_DIR / "outputs"
TEMP_DIR = WORK_DIR / "temp"
UPLOADED_DIR = WORK_DIR / "uploaded_audio"

# Expected repo layout when notebook runs from GitHub → Colab.
REPO_ROOT = Path(os.getenv("AMT_DEMO_REPO_ROOT", "/content/repo"))
DEMO_DIR = REPO_ROOT / "demo"
ASSET_DIR = REPO_ROOT / "demo_assets"
MANIFEST_PATH = DEMO_DIR / "sample_manifest.json"

# Optional subdirectories for saved outputs.
MIDI_DIR = OUTPUT_DIR / "midi"
PLOT_DIR = OUTPUT_DIR / "plots"
HTML_DIR = OUTPUT_DIR / "html"
AUDIO_DIR = OUTPUT_DIR / "audio"


def ensure_demo_dirs() -> None:
    for path in [
        WORK_DIR,
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        TEMP_DIR,
        UPLOADED_DIR,
        MIDI_DIR,
        PLOT_DIR,
        HTML_DIR,
        AUDIO_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
