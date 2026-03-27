# Piano AMT Pipeline — Complete Checklist

**Project:** Deep Learning Based Piano Transcription System with Chord and Note Reconstruction
**Supervisor:** Marcus Pearce, QMUL
**Dataset:** MAESTRO v3.0.0

---

## Section 1: Repository Structure

```
piano_amt/
├── src/                           # Core pipeline source code
│   ├── __init__.py                # Package marker
│   ├── constants.py               # ALL hyperparameters with paper citations
│   ├── audio.py                   # Audio loading + log-mel spectrogram
│   ├── midi.py                    # MIDI loading + 4-head piano-roll encoding/decoding
│   ├── dataset.py                 # MAESTRODataset + NPZ caching + build_cache()
│   ├── transforms.py              # Data augmentation (pitch shift, masking, gain)
│   ├── dataloader.py              # DataLoader factory + collate + sliding_windows
│   └── utils/
│       ├── __init__.py            # Package marker
│       └── viz.py                 # Visualisation: mel, piano roll, alignment check
├── notebooks/
│   ├── 00_setup_and_install.ipynb # GPU check, Drive mount, pip install, repo clone
│   ├── 01_download_maestro.ipynb  # Download + extract MAESTRO v3 to Drive
│   ├── 02_build_cache.ipynb       # Preprocess all files → NPZ cache on Drive
│   ├── 03_verify_pipeline.ipynb   # Session-start verification (5 checks)
│   └── 04_explore_data.ipynb      # Data exploration + alignment check + statistics
├── scripts/
│   └── verify_pipeline.py         # CLI: 5 asserted shape checks → "ALL CHECKS PASSED ✓"
├── configs/
│   └── default.yaml               # All hyperparameters in YAML format
├── checkpoints/                   # Auto-created by train.py; .gitkeep preserves dir
│   └── .gitkeep
├── train.py                       # OnsetsFramesLoss + Trainer + DummyModel + CLI
├── requirements.txt               # pip dependencies
├── CHECKLIST.md                   # This file
└── README.md                      # Project overview and quick-start guide
```

---

## Section 2: Research Provenance

| File | Constant / Function | Paper | Exact Justification |
|------|--------------------|----|---------------------|
| `src/constants.py` | `SAMPLE_RATE = 16000` | Hawthorne 2018a §3 Table 1 | Audio sampling rate used throughout |
| `src/constants.py` | `HOP_LENGTH = 512` | Hawthorne 2018a §3 Table 1 | Yields 31.25 frames/sec at 16 kHz |
| `src/constants.py` | `N_MELS = 229` | Hawthorne 2018a §3 Table 1 | Mel filterbank bins |
| `src/constants.py` | `MEL_FMIN = 30.0` | Hawthorne 2018a §3 Table 1 | Lowest mel frequency |
| `src/constants.py` | `MEL_FMAX = 8000.0` | Hawthorne 2018a §3 Table 1 | Highest mel frequency |
| `src/constants.py` | `FRAMES_PER_SECOND = 31.25` | Hawthorne 2018a §3 | Derived: 16000/512 |
| `src/constants.py` | `MIN_MIDI = 21, MAX_MIDI = 108` | Hawthorne 2018a §3 | Standard 88-key piano range |
| `src/constants.py` | `ONSET_WINDOW_FRAMES = 1` | Hawthorne 2018a §3.1 | Onset label width |
| `src/constants.py` | `OFFSET_WINDOW_FRAMES = 1` | Kim et al. 2025 D3RM §3 | Offset head for D3RM refiner target |
| `src/constants.py` | `VELOCITY_SCALE = 128.0` | Hawthorne 2018a §3.1 | velocity/128 normalisation |
| `src/constants.py` | `MAX_SEGMENT_FRAMES = 640` | jongwook/onsets-and-frames src/dataset.py | 640-frame crop = ~20s |
| `src/constants.py` | `MAX_SEGMENT_SAMPLES = 327680` | jongwook/onsets-and-frames src/dataset.py | 640 × 512 |
| `src/constants.py` | `MAESTRO_SPLIT_COL = "split"` | Hawthorne 2018b MAESTRO §3 | CSV column name |
| `src/audio.py` | `log(mel + 1e-9)` | jongwook src/mel.py line 27 | Log compression formula |
| `src/audio.py` | `T.MelSpectrogram(...)` | Hawthorne 2018a §3 Table 1 | All mel params |
| `src/midi.py` | `note_events_to_rolls()` | Hawthorne 2018a §3.1 | 4-head label encoding scheme |
| `src/midi.py` | offset head included | Kim et al. 2025 D3RM arxiv 2501.05068 | Offset errors are D3RM's primary target |
| `src/dataset.py` | NPZ caching strategy | jongwook/onsets-and-frames src/dataset.py | Precompute+cache for speed |
| `src/dataset.py` | 640-frame random crop | jongwook/onsets-and-frames src/dataset.py | MAX_SEGMENT_FRAMES crop |
| `src/dataset.py` | split column "split" | Hawthorne 2018b MAESTRO §3 | train/validation/test split |
| `src/transforms.py` | `RandomPitchShift(max_shift=1)` | KinWaiCheuk/ICPR2020 | ±1 semitone shift on mel + labels |
| `src/transforms.py` | `BINS_PER_SEMITONE = N_MELS/N_KEYS` | Hawthorne 2018a §3 | mel bin ↔ semitone mapping |
| `src/transforms.py` | `RandomTimeMask(max_mask_frames=50)` | KinWaiCheuk/ICPR2020 | SpecAugment-style time masking |
| `src/transforms.py` | `RandomFreqMask(max_mask_bins=20)` | KinWaiCheuk/ICPR2020 | SpecAugment-style freq masking |
| `src/dataloader.py` | `batch_size=8` (default config) | jongwook/onsets-and-frames src/train.py | Reference training config |
| `src/dataloader.py` | `num_workers=2` | jongwook/onsets-and-frames src/train.py | Reference training config |
| `train.py` | `OnsetsFramesLoss` with `pos_weight` | Hawthorne 2018a §3.2 | Weighted BCE for class imbalance |
| `train.py` | `Adam(lr=6e-4)` | Hawthorne 2018a §3.2 | Exact optimizer settings |
| `train.py` | `max_grad_norm=3.0` | Hawthorne 2018a §3.2 | Gradient clipping value |
| `train.py` | velocity masked MSE | Hawthorne 2018a §3.2 | Only compute velocity loss at onsets |

---

## Section 3: One-Time Setup Checklist (do once, never again)

### [ ] Step 1 — Open `00_setup_and_install.ipynb`

**Action:** Install packages, mount Drive, clone repo, verify imports.

**Expected time:** 3–5 minutes.

**What it produces:**
- Python packages installed in Colab runtime.
- Google Drive mounted at `/content/drive`.
- Repo at `/content/piano_amt`.

**How to verify it worked:**
```python
from src.constants import N_MELS, FRAMES_PER_SECOND, N_KEYS
assert N_MELS == 229
assert abs(FRAMES_PER_SECOND - 31.25) < 1e-6
print("✓ Setup OK")
```

---

### [ ] Step 2 — Open `01_download_maestro.ipynb`

**Action:** Download MAESTRO v3.0.0 ZIP (~16 GB) from Google Magenta storage
and extract to `/content/drive/MyDrive/piano_amt/maestro-v3.0.0/`.

**Expected time:** 15–45 minutes (download speed varies).

**What it produces:**
- `maestro-v3.0.0/` folder on Drive containing:
  - `maestro-v3.0.0.csv` (manifest of all audio+MIDI pairs)
  - Audio files (`.wav`, `.flac`)
  - MIDI files (`.midi`)

**How to verify it worked:**
```python
import pandas as pd, glob
csv = sorted(glob.glob("/content/drive/MyDrive/piano_amt/maestro-v3.0.0/*.csv"))[0]
df = pd.read_csv(csv)
assert len(df[df['split']=='train']) > 900  # MAESTRO v3 has 967 train files
print(f"✓ {len(df)} files found")
```

---

### [ ] Step 3 — Open `02_build_cache.ipynb`

**Action:** Preprocess every audio+MIDI pair → log-mel spectrogram + 4-head piano
rolls → save as compressed `.npz` to `/content/drive/MyDrive/piano_amt/cache/`.

**Expected time:** 25–40 minutes for the full dataset on a T4 GPU.

**What it produces:**
- One `.npz` file per piece in `cache/` (e.g., `2004_MIDI-Unprocessed_XP_22_R1_2004_01_ORIG_MID--AUDIO_22_R1_2004_12.npz`)
- Each NPZ contains: `mel (229,T)`, `onset (T,88)`, `frame (T,88)`, `offset (T,88)`, `velocity (T,88)`, `sr`
- Total cache size: ~12–15 GB

**How to verify it worked:**
```python
import glob, numpy as np
npzs = glob.glob("/content/drive/MyDrive/piano_amt/cache/*.npz")
print(f"Cache files: {len(npzs)}")  # expect ~1200
data = np.load(npzs[0])
assert data['mel'].shape[0] == 229
assert data['onset'].shape[1] == 88
print("✓ Cache verified")
```

---

## Section 4: Per-Session Checklist (every new Colab session)

### [ ] 1. Mount Drive (30 sec)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### [ ] 2. Clone/upload repo + set sys.path (1 min)
```python
import sys, os
REPO_DIR = '/content/piano_amt'
if not os.path.exists(REPO_DIR):
    os.system(f'git clone https://github.com/YOUR_USERNAME/piano_amt {REPO_DIR}')
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
```

### [ ] 3. Run `03_verify_pipeline.ipynb` — confirm all 5 checks pass (2 min)

Open `03_verify_pipeline.ipynb` and run all cells.  All 5 checks must print `✓ OK`.

Or run the script directly:
```bash
python /content/piano_amt/scripts/verify_pipeline.py \
    --maestro_root /content/drive/MyDrive/piano_amt/maestro-v3.0.0 \
    --max_files 3
```
Expected output: `ALL CHECKS PASSED ✓`

### [ ] 4. Ready to train

```bash
python /content/piano_amt/train.py \
    --maestro_root /content/drive/MyDrive/piano_amt/maestro-v3.0.0 \
    --cache_dir    /content/drive/MyDrive/piano_amt/cache \
    --checkpoint_dir /content/drive/MyDrive/piano_amt/checkpoints \
    --batch_size 8 \
    --epochs 30 \
    --num_workers 2
```

---

## Section 5: Data Flow Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MAESTRO v3.0.0                              │
│    maestro-v3.0.0.csv  +  *.wav  +  *.midi                         │
└───────────────────┬─────────────────────┬───────────────────────────┘
                    │                     │
                    ▼                     ▼
          ┌─────────────────┐   ┌──────────────────────┐
          │  src/audio.py   │   │   src/midi.py         │
          │  load_audio()   │   │   load_midi()         │
          │  wav_to_log_mel │   │   midi_to_note_events │
          │  → (229, T)     │   │   note_events_to_rolls│
          └────────┬────────┘   └──────────┬────────────┘
                   │                        │
                   │  log-mel (229,T)       │  4-head rolls (T,88)×4
                   │                        │
                   └──────────┬─────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   src/dataset.py      │
                    │   preprocess_and_cache│
                    │   → saves .npz        │
                    │   (mel, onset, frame, │
                    │    offset, velocity)  │
                    └──────────┬───────────┘
                               │
                    /content/drive/MyDrive/piano_amt/cache/
                               │
                               ▼
                    ┌──────────────────────┐
                    │   MAESTRODataset      │
                    │   __getitem__()       │
                    │   load_from_cache()   │
                    │   _random_segment()   │
                    │   → Dict (229,640)    │
                    │     (640,88)×4        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  src/transforms.py    │ ← training only
                    │  _AugmentedDataset    │
                    │  RandomPitchShift     │
                    │  RandomTimeMask       │
                    │  RandomFreqMask       │
                    │  RandomGainJitter     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  src/dataloader.py    │
                    │  piano_amt_collate()  │
                    │  → batch Dict         │
                    │    mel:     (B,229,T) │
                    │    onset:   (B,T,88)  │
                    │    frame:   (B,T,88)  │
                    │    offset:  (B,T,88)  │
                    │    velocity:(B,T,88)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      train.py         │
                    │  OnsetsFramesLoss     │
                    │  Trainer.train_epoch  │
                    │  → checkpoints/*.pt   │
                    └──────────────────────┘
```

---

## Section 6: Shapes Reference

| Pipeline Stage | Variable | Shape | Notes |
|----------------|----------|-------|-------|
| Raw audio | `waveform` | `(1, N_samples)` | Mono, 16 kHz |
| Log-mel (full piece) | `log_mel` | `(229, T_full)` | T ≈ 31.25 × duration_sec |
| Log-mel (segment) | `mel` | `(229, 640)` | Random crop, training |
| Onset roll (full) | `onset` | `(T_full, 88)` | float32, values ∈ {0, 1} |
| Frame roll (full) | `frame` | `(T_full, 88)` | float32, values ∈ {0, 1} |
| Offset roll (full) | `offset` | `(T_full, 88)` | float32, values ∈ {0, 1} |
| Velocity roll (full) | `velocity` | `(T_full, 88)` | float32, values ∈ [0, 1] |
| Dataset item (segmented) | `item["mel"]` | `(229, 640)` | After `_random_segment` |
| Dataset item (roll) | `item["onset"]` | `(640, 88)` | After `_random_segment` |
| Batch (mel) | `batch["mel"]` | `(B, 229, 640)` | After collation |
| Batch (roll) | `batch["onset"]` | `(B, 640, 88)` | After collation |
| Model output | `pred["onset"]` | `(B, T, 88)` | Logits (pre-sigmoid) |
| NPZ file: mel | `data["mel"]` | `(229, T_full)` | Saved as float32 |
| NPZ file: onset | `data["onset"]` | `(T_full, 88)` | Saved as float32 |
| Sliding window (inference) | `window["mel"]` | `(229, 640)` | Zero-padded last window |

---

## Section 7: Model Plug-In Instructions

### Current state
`train.py` uses `_DummyModel` which returns zeros for all 4 heads.
The full training loop (loss, backward, grad clip, optimizer, checkpointing) is
functional and validated.

### Steps to plug in the real OnsetsAndFrames model

1. **Implement your model** in a new file, e.g. `src/model.py`:
   ```python
   class OnsetsAndFrames(nn.Module):
       def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
           # mel: (B, 229, T)
           # must return: {"onset": (B,T,88), "frame": (B,T,88),
           #               "offset": (B,T,88), "velocity": (B,T,88)}
           # All values are raw logits (pre-sigmoid)
           ...
   ```

2. **Import and instantiate** in `train.py` `main()`:
   ```python
   from src.model import OnsetsAndFrames
   model = OnsetsAndFrames()  # replace _DummyModel()
   ```

3. **Model interface contract:**
   - Input: `mel` — FloatTensor `(B, 229, 640)` — log-mel spectrogram batch
   - Output: Dict with exactly 4 keys:
     - `"onset"`: FloatTensor `(B, 640, 88)` — raw logits
     - `"frame"`: FloatTensor `(B, 640, 88)` — raw logits
     - `"offset"`: FloatTensor `(B, 640, 88)` — raw logits
     - `"velocity"`: FloatTensor `(B, 640, 88)` — values in [0,1] (no sigmoid needed in loss for MSE)
   - No other changes needed — `OnsetsFramesLoss` and `Trainer` are model-agnostic.

4. **Inference** (after training):
   ```python
   from src.dataloader import sliding_windows
   from src.midi import rolls_to_midi

   log_mel = load_audio_as_log_mel(audio_path)  # (229, T)
   windows = sliding_windows(log_mel)            # List[Dict]
   # Run model on each window, stitch predictions, decode with rolls_to_midi()
   ```

---

## Section 8: Troubleshooting

### Out-of-Memory (OOM) on GPU
**Symptom:** `RuntimeError: CUDA out of memory`
**Fix:** Reduce `--batch_size` (try 4 → 2 → 1).  Also reduce `--num_workers` to 0.

```bash
python train.py --maestro_root ... --batch_size 2 --num_workers 0
```

---

### FileNotFoundError: Cache not built
**Symptom:** `FileNotFoundError: /content/drive/.../cache/xxx.npz`
**Fix:** The NPZ cache hasn't been built yet. Run `02_build_cache.ipynb` first.

---

### Misaligned labels (alignment check fails)
**Symptom:** In `plot_mel_with_labels()`, bright mel bands don't line up with frame roll.
**Fix:**
- Check `FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH = 16000 / 512 = 31.25`
- Check `midi_path_to_rolls()` uses `fps=FRAMES_PER_SECOND`
- Ensure audio was resampled to 16 kHz before computing mel
- Ensure `start_sec` offset is correctly subtracted in `midi_path_to_rolls()`

---

### Drive disconnected mid-session
**Symptom:** `OSError: [Errno 5] Input/output error` on Drive files
**Fix:** Re-mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

### ImportError: No module named 'src'
**Symptom:** `ModuleNotFoundError: No module named 'src'`
**Fix:** `sys.path` is not set up. Run:
```python
import sys
sys.path.insert(0, '/content/piano_amt')
from src.constants import N_MELS  # test
```

---

### pretty_midi MIDI parse error
**Symptom:** `Exception: MIDI file has no tracks` or similar
**Fix:** The MIDI file may be corrupted. Skip it in `build_cache` (already handled by
`try/except` in `build_cache()`). Check error count in output.

---

### Slow DataLoader
**Symptom:** GPU utilisation is low; DataLoader is the bottleneck
**Fix:**
- Ensure cache is on Drive (not raw audio loading)
- Increase `num_workers` (try 4 on Colab Pro)
- Set `pin_memory=True` (already default)
- If Drive is slow, copy cache to local `/content/` at session start:
  ```bash
  cp -r /content/drive/MyDrive/piano_amt/cache /content/cache
  ```

---

*Generated for Piano AMT dissertation project — QMUL, supervised by Marcus Pearce.*
