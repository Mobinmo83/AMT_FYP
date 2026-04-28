from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hugging Face settings
# ---------------------------------------------------------------------------
HF_REPO_ID = os.getenv("AMT_DEMO_HF_REPO_ID", "Mobinmo83/piano-amt-demo")
HF_CHECKPOINT_FILENAME = os.getenv("AMT_DEMO_HF_CHECKPOINT_FILENAME", "checkpoints/best.pt")
HF_REPO_TYPE = os.getenv("AMT_DEMO_HF_REPO_TYPE", "model")

# ---------------------------------------------------------------------------
# Model + audio settings
# ---------------------------------------------------------------------------
MODEL_COMPLEXITY = int(os.getenv("AMT_DEMO_MODEL_COMPLEXITY", "48"))
SAMPLE_RATE = int(os.getenv("AMT_DEMO_SAMPLE_RATE", "16000"))

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

MIDI_DIR = OUTPUT_DIR / "midi"
PLOT_DIR = OUTPUT_DIR / "plots"
HTML_DIR = OUTPUT_DIR / "html"
AUDIO_DIR = OUTPUT_DIR / "audio"

# FluidR3 is installed by ``apt-get install fluid-soundfont-gm`` on Colab.
DEFAULT_SF2_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
]


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
