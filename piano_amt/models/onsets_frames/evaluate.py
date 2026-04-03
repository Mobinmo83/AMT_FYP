"""
models/onsets_frames/evaluate.py — Evaluation harness for OnsetsAndFrames.

Runs the model on the validation or test split, computes all AMT metrics via
evaluate/metrics.py, saves per-file results, summary JSON, and piano-roll plots.

Evaluation strategy:
    Matches both Magenta and jongwook/onsets-and-frames exactly:
    each piece is processed as a SINGLE full-length forward pass
    (no windowing, no chunking), giving the BiLSTM complete
    bidirectional context over the entire performance.

    Magenta README: "validation/test examples containing full pieces"
    jongwook evaluate.py: sequence_length=None → full piece in one pass
    Hawthorne 2018a §4: evaluation on complete test recordings

    Requires A100 (40/80GB) or H100 (80GB) for longest MAESTRO pieces.
    Not suitable for T4 (16GB) on pieces longer than ~4 minutes.

Usage:
    python -m models.onsets_frames.evaluate \\
        --checkpoint  /content/drive/MyDrive/piano_amt/runs/of_baseline_20h/checkpoints/best.pt \\
        --maestro_root /content/drive/MyDrive/piano_amt/maestro-v3.0.0 \\
        --cache_dir    /content/drive/MyDrive/piano_amt/cache \\
        --split test \\
        --max_files 20 \\
        --save_midi \\
        --save_plots

Outputs (written into the run directory beside the checkpoint):
    eval_<split>/
        summary_metrics.json     ← mean F1 scores across all files
        per_file_metrics.json    ← per-file breakdown
        plots/                   ← piano-roll comparison images (if --save_plots)
        midi_samples/            ← decoded MIDI files (if --save_midi)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.onsets_frames.model import OnsetsAndFrames
from models.onsets_frames.decode import rolls_to_note_events
from src.constants import N_KEYS, N_MELS, FRAMES_PER_SECOND, MAX_SEGMENT_FRAMES
from src.dataloader import get_dataloader, sliding_windows
from src.audio import load_audio_as_log_mel
from src.dataset import MAESTRODataset, load_from_cache, _cache_path
from src.midi import rolls_to_midi
from evaluate.metrics import compute_metrics
from evaluate.error_analysis import compute_error_analysis
from evaluate.plots import plot_piano_roll_comparison


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_file(
    model:      OnsetsAndFrames,
    cache_path: Path,
    device:     torch.device,
    onset_threshold:  float = 0.5,
    frame_threshold:  float = 0.5,
    offset_threshold: float = 0.5,
) -> Dict:
    """
    Run model on one full-length cached file and compute all metrics.

    The entire mel spectrogram is passed through the model in a single
    forward pass — no windowing or chunking. This matches exactly how
    both Magenta and jongwook evaluate:

      - Magenta: test TFRecords contain "full pieces" (README), processed
        by the TF Estimator in one pass through the acoustic model + BiLSTMs.
      - jongwook: evaluate.py calls model.run_on_batch(label) where label
        contains the full piece (sequence_length=None at eval time).

    Why this matters: the model has 4 BiLSTMs (onset, offset, frame combined,
    velocity has none). BiLSTMs build context left-to-right and right-to-left.
    Full-length inference gives every frame complete bidirectional context
    over the entire performance. Chunked inference resets hidden states at
    boundaries, degrading predictions at chunk edges.

    Requires GPU with sufficient VRAM (A100 40GB+ or H100 80GB).

    Returns:
        Dict with keys: all metrics from compute_metrics(), error_analysis,
                        pred tensors for downstream use (prefixed with _)
    """
    data = load_from_cache(cache_path)
    mel  = data["mel"]          # (229, T_full)

    # Ground-truth rolls
    gt_onset    = data["onset"]     # (T_full, 88)
    gt_frame    = data["frame"]
    gt_offset   = data["offset"]
    gt_velocity = data["velocity"]

    T_full = mel.shape[1]

    # # --- Full-length single-pass inference (Magenta + jongwook approach) ---
    # model.eval()
    # with torch.no_grad():
    #     # (229, T_full) → (1, 229, T_full) — single batch, full piece
    #     w_mel = mel.unsqueeze(0).to(device) 
    #     out   = model(w_mel)

    #     pred_onset    = out["onset"][0].cpu()      # (T_full, 88)
    #     pred_frame    = out["frame"][0].cpu()
    #     pred_offset   = out["offset"][0].cpu()
    #     pred_velocity = out["velocity"][0].cpu()
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        w_mel = mel.unsqueeze(0).to(device)
        out   = model(w_mel)

        pred_onset    = out["onset"][0].cpu()
        pred_frame    = out["frame"][0].cpu()
        pred_offset   = out["offset"][0].cpu()
        pred_velocity = out["velocity"][0].cpu()

    # Compute note-level and frame-level metrics
    metrics = compute_metrics(
        pred_onset=pred_onset,
        pred_frame=pred_frame,
        pred_offset=pred_offset,
        pred_velocity=pred_velocity,
        gt_onset=gt_onset,
        gt_frame=gt_frame,
        gt_offset=gt_offset,
        gt_velocity=gt_velocity,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        offset_threshold=offset_threshold,
        fps=FRAMES_PER_SECOND,
    )

    # Error analysis
    ea = compute_error_analysis(
        pred_onset=pred_onset,
        pred_frame=pred_frame,
        pred_offset=pred_offset,
        gt_onset=gt_onset,
        gt_frame=gt_frame,
        gt_offset=gt_offset,
        onset_threshold=onset_threshold,
        fps=FRAMES_PER_SECOND,
    )
    metrics["error_analysis"] = ea

    # Attach predictions for optional downstream use
    metrics["_pred_onset"]    = pred_onset
    metrics["_pred_frame"]    = pred_frame
    metrics["_pred_offset"]   = pred_offset
    metrics["_pred_velocity"] = pred_velocity
    metrics["_gt_frame"]      = gt_frame
    metrics["_gt_onset"]      = gt_onset

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str | Path,
    maestro_root:    str | Path,
    cache_dir:       str | Path,
    split:           str = "test",
    max_files:       Optional[int] = None,
    save_midi:       bool = False,
    save_plots:      bool = False,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    model_complexity: int = 48,
) -> Dict:
    """
    Full evaluation run. Returns summary metrics dict.

    Outputs are written into:
        <checkpoint_dir>/../eval_<split>/
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = OnsetsAndFrames(model_complexity=model_complexity)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {ckpt.get('epoch','?')} epochs, "
          f"val_loss={ckpt.get('val_loss',0):.4f}")

    # Output directory
    ckpt_dir = Path(checkpoint_path).parent
    run_dir  = ckpt_dir.parent
    eval_dir = run_dir / f"eval_{split}"
    eval_dir.mkdir(exist_ok=True)
    plots_dir = eval_dir / "plots"
    midi_dir  = eval_dir / "midi_samples"
    if save_plots: plots_dir.mkdir(exist_ok=True)
    if save_midi:  midi_dir.mkdir(exist_ok=True)

    # Collect files for this split
    import pandas as pd, glob as glob_mod
    csv_files = sorted(Path(maestro_root).glob("*.csv"))
    df = pd.read_csv(csv_files[0])
    split_df = df[df["split"] == split].reset_index(drop=True)
    if max_files:
        split_df = split_df.head(max_files)

    print(f"\nEvaluating {len(split_df)} files from '{split}' split...")
    print(f" full length single pass inference")

    all_metrics: List[Dict] = []
    per_file:    List[Dict] = []


    skipped = 0

    for i, row in split_df.iterrows():
        audio_path = str(Path(maestro_root) / row["audio_filename"])
        stem       = Path(audio_path).stem
        cp         = _cache_path(audio_path, cache_dir)

        if not cp.exists():
            skipped += 1
            continue

        file_metrics = evaluate_file(
            model, cp, device,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )

        # Collect scalar metrics for averaging
        scalars = {k: v for k, v in file_metrics.items() if not k.startswith("_")}
        # Flatten error_analysis
        ea = scalars.pop("error_analysis", {})
        scalars.update({f"ea_{k}": v for k, v in ea.items()})
        scalars["stem"] = stem
        per_file.append(scalars)
        all_metrics.append(scalars)

        # Optional: save plots
        if save_plots and i < 10:   # first 10 files only
            plot_piano_roll_comparison(
                pred_frame=file_metrics["_pred_frame"],
                gt_frame=file_metrics["_gt_frame"],
                title=stem,
                save_path=plots_dir / f"{stem}.png",
            )

        # Optional: save MIDI
        if save_midi and i < 20:    # first 20 files
            pm = rolls_to_midi(
                onset_roll=file_metrics["_pred_onset"],
                frame_roll=file_metrics["_pred_frame"],
                velocity_roll=file_metrics["_pred_velocity"],
                fps=FRAMES_PER_SECOND,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
            )
            pm.write(str(midi_dir / f"{stem}.mid"))

    # Compute summary (mean across files)
    total_requested = len(split_df)
    total_evaluated = len(all_metrics)
    print(f"\n  ✓ Successfully evaluated {total_evaluated}/{total_requested} files"
          + (f" ({skipped} skipped — no cache)" if skipped > 0 else ""))

    # Compute summary (mean across files)
    if not all_metrics:
        print("No files evaluated.")
        return {}

    summary = {}
    numeric_keys = [k for k in all_metrics[0] if k != "stem" and
                    isinstance(all_metrics[0][k], (int, float))]
    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        summary[k] = float(np.mean(vals)) if vals else 0.0

    summary["n_files"]    = len(all_metrics)
    summary["split"]      = split
    summary["checkpoint"] = str(checkpoint_path)

    # Save outputs
    with open(eval_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_dir / "per_file_metrics.json", "w") as f:
        json.dump(per_file, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SUMMARY ({split}, n={summary['n_files']})")
    print(f"{'='*50}")
    for k in ["onset_f1", "frame_f1", "note_with_offset_f1", "note_with_offset_vel_f1"]:
        print(f"  {k:30s}: {summary.get(k, 0):.4f}")
    print(f"\nResults saved → {eval_dir}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OnsetsAndFrames checkpoint."
    )
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--maestro_root",    required=True)
    parser.add_argument("--cache_dir",       default=None)
    parser.add_argument("--split",           default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max_files",       type=int, default=None)
    parser.add_argument("--save_midi",       action="store_true")
    parser.add_argument("--save_plots",      action="store_true")
    parser.add_argument("--onset_threshold", type=float, default=0.5)
    parser.add_argument("--frame_threshold", type=float, default=0.5)
    parser.add_argument("--model_complexity",type=int,   default=48)
    args = parser.parse_args()

    cache_dir = args.cache_dir or str(Path(args.maestro_root) / "cache")

    run_evaluation(
        checkpoint_path=args.checkpoint,
        maestro_root=args.maestro_root,
        cache_dir=cache_dir,
        split=args.split,
        max_files=args.max_files,
        save_midi=args.save_midi,
        save_plots=args.save_plots,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        model_complexity=args.model_complexity,
    )


if __name__ == "__main__":
    main()