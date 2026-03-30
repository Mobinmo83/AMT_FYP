# Piano AMT Pipeline — Complete Checklist

**Project:** Deep Learning Based Piano Transcription System with Chord and Note Reconstruction
**Supervisor:** Marcus Pearce, QMUL
**Dataset:** MAESTRO v3.0.0

---

## Section 1: Repository Structure

piano_amt/
├── src/                           # Core pipeline source code
│   ├── __init__.py
│   ├── constants.py               # ALL hyperparameters with paper citations
│   ├── audio.py                   # Audio loading + log-mel spectrogram (torchaudio)
│   ├── midi.py                    # MIDI loading + 4-head piano-roll encoding/decoding
│   ├── dataset.py                 # MAESTRODataset + NPZ caching + build_cache()
│   ├── transforms.py              # Data augmentation (pitch shift, masking, gain)
│   ├── dataloader.py              # DataLoader factory + collate + sliding_windows
│   └── utils/
│       ├── __init__.py
│       └── viz.py                 # Visualisation: mel, piano roll, alignment check
├── models/
│   ├── __init__.py
│   ├── onsets_frames/
│   │   ├── __init__.py
│   │   ├── model.py               # OnsetsAndFrames (jongwook architecture, 26.5M params)
│   │   ├── model_adv.py           # Adversarial variant (placeholder)
│   │   ├── train.py               # Loss + Trainer + AMP + checkpointing + CLI
│   │   ├── evaluate.py            # Full evaluation harness (metrics + plots + MIDI)
│   │   └── decode.py              # Piano-roll → note events / MIDI decoding
│   └── evaluate/                  # Shared evaluation utilities
│       ├── __init__.py
│       ├── metrics.py             # Frame F1, Onset F1, Note+Offset F1 (mir_eval)
│       ├── error_analysis.py      # Duplicates, chord completeness, offset MAE
│       ├── compare.py             # Cross-run comparison: CSV, LaTeX, bar charts
│       └── plots.py               # Training curves, piano-roll comparison
├── notebooks/
│   ├── dataset_setup_install_download_cache_verify.ipynb  # Data pipeline (setup → cache → verify)
│   ├── 05_train_onsets_frames.ipynb                       # Model training (smoke → full)
│   └── data_exploring.ipynb                               # Data exploration & visualisation
├── scripts/
│   └── verify_pipeline.py         # CLI: 5 asserted shape checks
├── configs/
│   └── default.yaml               # All hyperparameters in YAML format
├── requirements.txt
├── CHECKLIST.md                   # This file
└── README.md

---

## Section 2: Research Provenance

### Audio & Preprocessing (from Hawthorne 2018a §3 + jongwook)

| File | Constant / Function | Source | Justification |
|------|---------------------|--------|---------------|
| constants.py | SAMPLE_RATE = 16000 | Hawthorne 2018a §3 Table 1 | Audio sampling rate |
| constants.py | HOP_LENGTH = 512 | Hawthorne 2018a §3 Table 1 | 31.25 fps at 16 kHz |
| constants.py | N_MELS = 229 | Hawthorne 2018a §3 Table 1 | Mel filterbank bins |
| constants.py | MEL_FMIN = 30.0 | Hawthorne 2018a §3 Table 1 | Lowest mel frequency |
| constants.py | MEL_FMAX = 8000.0 | Hawthorne 2018a §3 Table 1 | Highest mel frequency |
| constants.py | MIN_MIDI=21, MAX_MIDI=108 | Hawthorne 2018a §3 | 88-key piano range |
| audio.py | log(mel + 1e-9) | jongwook src/mel.py line 27 | Log compression formula |
| audio.py | T.MelSpectrogram(...) | Hawthorne 2018a §3 Table 1 | All mel parameters |
| midi.py | note_events_to_rolls() | Hawthorne 2018a §3.1 | 4-head label encoding |
| midi.py | offset head | jongwook/onsets-and-frames | Offset head improvement |

### Dataset & Training (from jongwook + KinWaiCheuk)

| File | Constant / Function | Source | Justification |
|------|---------------------|--------|---------------|
| constants.py | MAX_SEGMENT_FRAMES=640 | jongwook src/dataset.py | ~20s random crop |
| dataset.py | NPZ caching strategy | jongwook src/dataset.py | Precompute + cache |
| dataset.py | split column "split" | Hawthorne 2018b §3 | MAESTRO train/val/test |
| transforms.py | RandomPitchShift(±1) | KinWaiCheuk/ICPR2020 | Mel + label joint shift |
| transforms.py | RandomTimeMask(50) | KinWaiCheuk/ICPR2020 | SpecAugment time mask |
| transforms.py | RandomFreqMask(20) | KinWaiCheuk/ICPR2020 | SpecAugment freq mask |
| dataloader.py | batch_size=8 | jongwook src/train.py | Reference config |
| train.py | Adam(lr=6e-4) | Hawthorne 2018a §3.2 | Exact optimizer settings |
| train.py | max_grad_norm=3.0 | Hawthorne 2018a §3.2 | Per-param gradient clip |
| train.py | velocity masked MSE | Hawthorne 2018a §3.2 | Only at onset frames |

### Model Architecture (from jongwook transcriber.py)

| Component | Source | Detail |
|-----------|--------|--------|
| ConvStack | jongwook transcriber.py lines 12–42 | 3 conv layers, 2 MaxPool, 3 Dropout |
| Channel counts | jongwook | out//16, out//16, out//8 (48, 48, 96 at complexity=48) |
| BiLSTM | jongwook lstm.py | hidden=output_size//2 per direction |
| onset/offset stacks | jongwook lines 57–67 | ConvStack → BiLSTM → Linear → Sigmoid |
| frame_stack | jongwook line 69 | ConvStack → Linear → Sigmoid (no BiLSTM) |
| combined_stack | jongwook line 75 | BiLSTM on onset⊕offset⊕frame (dim×3) |
| velocity_stack | jongwook line 80 | ConvStack → Linear (NO sigmoid) |
| Gradient stopping | jongwook line 92 | onset.detach() + offset.detach() |
| model_complexity=48 | jongwook | model_size = 48×16 = 768, ~26.5M params |

### Pipeline Customisations (original to this project)

| Customisation | Reference baseline | What changed | Why |
|---------------|-------------------|-------------|-----|
| Weighted BCE (pos_weight=5.0) | jongwook uses unweighted | Adds positive class weight | Addresses ~97% negative class imbalance in piano rolls |
| Data augmentation | jongwook has none | Added KinWaiCheuk pitch shift + SpecAugment | Improved generalisation |
| ReduceLROnPlateau | jongwook uses fixed LR | Halves LR after 3 stale epochs | Standard practice for better convergence |
| NPZ pre-caching | jongwook computes mel on-the-fly | Pre-cache all mel + rolls to Drive | Faster on Colab where Drive I/O is bottleneck |
| AMP + torch.compile | Not in jongwook | float16 forward + kernel fusion | ~2× speed on T4, ~4× on A100 |
| Every-epoch checkpointing | jongwook saves best only | latest.pt + best.pt every epoch | Colab crash safety |
| RandomGainJitter | Not in any reference | ±3dB additive in log space | Simulates recording-level variation |

---

## Section 3: Notebooks Guide

| Notebook | Purpose | When to run |
|----------|---------|------------|
| dataset_setup_install_download_cache_verify.ipynb | GPU check, Drive mount, install, clone, download MAESTRO, build NPZ cache, verify pipeline (5 checks) | Data pipeline: once for download/cache, verify section every session |
| train_onsets_frames.ipynb | Model training with staged scaling (smoke→tiny→small→medium→full), evaluation, comparison, MIDI demo | Every training session |
| data_exploring_verification.ipynb | Data exploration: mel spectrograms, piano rolls, alignment checks, dataset statistics | Optional — for inspection and report figures |

---

## Section 4: Model Interface Contract

The OnsetsAndFrames model follows this exact interface:

**Input:** `mel` — FloatTensor `(B, 229, 640)` — log-mel spectrogram batch

**Output:** Dict with exactly 4 keys:
- `"onset"`:    FloatTensor `(B, 640, 88)` — post-sigmoid probabilities [0,1]
- `"frame"`:    FloatTensor `(B, 640, 88)` — post-sigmoid probabilities [0,1]
- `"offset"`:   FloatTensor `(B, 640, 88)` — post-sigmoid probabilities [0,1]
- `"velocity"`: FloatTensor `(B, 640, 88)` — raw output (no sigmoid)

**Loss compatibility:**
- onset/frame/offset use BCELoss (post-sigmoid → NOT BCEWithLogitsLoss)
- velocity uses masked MSE (raw output compared to velocity/128 targets)

---

## Section 5: Checkpoint Contents

Every `.pt` checkpoint saved by train.py contains:

| Key | Type | Purpose |
|-----|------|---------|
| model_state | OrderedDict | All 26.5M model weights (always fp32) |
| optimizer_state | dict | Adam momentum buffers + step counts |
| scheduler_state | dict | ReduceLROnPlateau patience counter + best |
| scaler_state | dict | AMP GradScaler scale factor (if AMP enabled) |
| epoch | int | Last completed epoch number |
| val_loss | float | Validation loss at this epoch |
| global_step | int | Total training steps completed |
| best_val_loss | float | Best val loss seen across all sessions |
| use_amp | bool | Whether AMP was used |

**Checkpoint files on Drive:**
- `latest.pt` — saved EVERY epoch (resume from here after disconnect)
- `best.pt` — saved only when val_loss improves (use for evaluation)
- `epoch_NNN_valloss_X.XXXX.pt` — named snapshots at each improvement

---

## Section 6: Shapes Reference

| Stage | Variable | Shape | Notes |
|-------|----------|-------|-------|
| Raw audio | waveform | (1, N_samples) | Mono, 16 kHz |
| Log-mel (full) | log_mel | (229, T_full) | T ≈ 31.25 × duration_sec |
| Log-mel (segment) | mel | (229, 640) | Random crop, training |
| Roll (full) | onset | (T_full, 88) | float32, binary |
| Roll (segment) | onset | (640, 88) | After _random_segment |
| Batch mel | batch["mel"] | (B, 229, 640) | After collation |
| Batch roll | batch["onset"] | (B, 640, 88) | After collation |
| Model output | pred["onset"] | (B, T, 88) | Post-sigmoid [0,1] |
| Model velocity | pred["velocity"] | (B, T, 88) | Raw (no sigmoid) |
| NPZ mel | data["mel"] | (229, T_full) | float32 |
| NPZ roll | data["onset"] | (T_full, 88) | float32 |

---

## Section 7: Training Stages

| Stage | max_files | complexity | epochs | batch_size | Purpose |
|-------|-----------|------------|--------|------------|---------|
| 0 smoke | 5 | 16 | 2 | 4 | Verify loop runs |
| 1 tiny | 30 | 16 | 5 | 4 | Check model learns |
| 2 small | 100 | 48 | 15 | 8 | Hyperparameter check |
| 3 medium | 400 | 48 | 30 | 8 | Dissertation comparison |
| 4 full | None | 48 | 30+ | 8 | Final results |

---

## Section 8: Troubleshooting

### Out-of-Memory (OOM)
Reduce batch_size first (8→4→2), NOT model_complexity.

### Colab disconnect mid-training
Resume with: --resume .../checkpoints/latest.pt
Everything is restored: model, optimizer, scheduler, AMP scaler, epoch, metrics history.

### ImportError: No module named 'src'
Set sys.path: sys.path.insert(0, '/content/AMT_FYP/piano_amt')

### Cache not built
Run the data pipeline notebook first (download + build_cache sections).

### Drive disconnected
Re-mount: drive.mount('/content/drive', force_remount=True)
