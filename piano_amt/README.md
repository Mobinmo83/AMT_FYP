# Piano AMT — Deep Learning Based Piano Transcription System

**Project:** Deep Learning Based Piano Transcription System with Chord and Note Reconstruction
**Supervisor:** Marcus Pearce, Queen Mary University of London (QMUL)
**Dataset:** MAESTRO v3.0.0

---

## Research Context

This system implements the **Onsets and Frames** dual-objective piano transcription
architecture (Hawthorne et al. 2018a) trained on the **MAESTRO** benchmark dataset
(Hawthorne et al. 2018b).  The preprocessing pipeline produces four-head piano-roll
labels — onset, frame, offset, and velocity — enabling future integration of the
**D3RM** diffusion-based refiner (Kim, Kwon & Nam 2025) which specifically targets
offset detection accuracy.  Data augmentation follows KinWaiCheuk/ICPR2020:
±1-semitone pitch shifting applied jointly to mel bins and label columns, plus
SpecAugment-style time and frequency masking for improved generalisation.

---

## Quick Start (Google Colab Pro, T4 GPU)

**Step 1 — Run once (download + preprocess):**
```
00_setup_and_install.ipynb  →  01_download_maestro.ipynb  →  02_build_cache.ipynb
```

**Step 2 — Run every session (verify):**
```
03_verify_pipeline.ipynb
```

**Step 3 — Train:**
```bash
python train.py \
    --maestro_root /content/drive/MyDrive/piano_amt/maestro-v3.0.0 \
    --cache_dir    /content/drive/MyDrive/piano_amt/cache \
    --checkpoint_dir /content/drive/MyDrive/piano_amt/checkpoints \
    --batch_size 8 --epochs 30 --num_workers 2
```

---

## Folder Structure

| Path | Description |
|------|-------------|
| `src/constants.py` | All hyperparameters with paper citations |
| `src/audio.py` | Audio loading + log-mel spectrogram (torchaudio) |
| `src/midi.py` | MIDI parsing + 4-head piano-roll encoding/decoding |
| `src/dataset.py` | MAESTRODataset + NPZ caching + build_cache() |
| `src/transforms.py` | Data augmentation (pitch shift, SpecAugment, gain) |
| `src/dataloader.py` | DataLoader factory + custom collate + sliding windows |
| `src/utils/viz.py` | Mel/roll visualisation + alignment check |
| `train.py` | OnsetsFramesLoss + Trainer + CLI |
| `scripts/verify_pipeline.py` | 5 asserted shape checks → ALL CHECKS PASSED ✓ |
| `configs/default.yaml` | Hyperparameter config file |
| `notebooks/` | Five Jupyter notebooks (setup → explore) |
| `CHECKLIST.md` | Complete step-by-step guide with paper provenance |

---

## Notebooks Guide

| Notebook | Purpose | When to run |
|----------|---------|------------|
| `00_setup_and_install.ipynb` | GPU check, Drive mount, pip install, repo clone | Every session |
| `01_download_maestro.ipynb` | Download MAESTRO v3 (~16 GB) to Drive | Once |
| `02_build_cache.ipynb` | Preprocess all audio+MIDI → NPZ cache (~30 min) | Once |
| `03_verify_pipeline.ipynb` | 5 pipeline checks (shapes, loading, DataLoader) | Every session |
| `04_explore_data.ipynb` | Statistics, mel viz, alignment check, augmentation | Optional |

---

## Pipeline Overview

```
.wav + .midi  →  log-mel (229,T) + piano rolls (T,88)  →  NPZ cache
                                                              ↓
                                              MAESTRODataset (640-frame crop)
                                                              ↓
                                              Augmentation (pitch shift, masking)
                                                              ↓
                                              DataLoader batch (B,229,640) + (B,640,88)×4
                                                              ↓
                                              OnsetsFramesLoss (BCE×3 + masked MSE)
```

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA T4 or better (16 GB VRAM recommended) |
| RAM | 12+ GB system RAM |
| Storage | ~16 GB Drive for MAESTRO + ~15 GB for NPZ cache |
| Runtime | Google Colab Pro (T4 GPU, high RAM) |

---

## Key Hyperparameters (Hawthorne 2018a §3)

| Parameter | Value | Source |
|-----------|-------|--------|
| Sample rate | 16 kHz | Hawthorne 2018a §3 |
| Hop length | 512 samples | Hawthorne 2018a §3 |
| Mel bins | 229 | Hawthorne 2018a §3 |
| Mel freq range | 30–8000 Hz | Hawthorne 2018a §3 |
| Frame rate | 31.25 fps | 16000/512 |
| Segment length | 640 frames (~20s) | jongwook/onsets-and-frames |
| Learning rate | 6e-4 (Adam) | Hawthorne 2018a §3.2 |
| Grad clip | 3.0 | Hawthorne 2018a §3.2 |
| Batch size | 8 | jongwook/onsets-and-frames |

---

## References

1. **Hawthorne et al. 2018a** — "Onsets and Frames: Dual-Objective Piano Transcription"
   https://arxiv.org/abs/1810.12247

2. **Hawthorne et al. 2018b** — "Enabling Factorized Piano Music Modeling and Generation
   with the MAESTRO Dataset" https://arxiv.org/abs/1810.12247

3. **jongwook/onsets-and-frames** — Reference PyTorch implementation
   https://github.com/jongwook/onsets-and-frames

4. **KinWaiCheuk/ICPR2020** — Augmentation strategy (pitch shift + SpecAugment)
   https://github.com/KinWaiCheuk/ICPR2020

5. **Kim, Kwon & Nam 2025** — "D3RM: A Discrete Denoising Diffusion Refiner for
   Music Transcription" https://arxiv.org/abs/2501.05068
