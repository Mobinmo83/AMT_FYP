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

    cuDNN is disabled for LSTM layers during evaluation to support
    arbitrarily long sequences (full MAESTRO pieces can exceed 50k frames).
    The native PyTorch LSTM kernel has no sequence length limit.

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
        summary_metrics.json     ← mean F1 scores across all files + protocol
        per_file_metrics.json    ← per-file breakdown
        plots/                   ← piano-roll comparison images (if --save_plots)
        midi_samples/            ← decoded MIDI files (if --save_midi)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from evaluate.metrics import compute_metrics, get_eval_protocol
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
    onset_tolerance:     float = 0.05,
    offset_ratio:        float = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance:  float = 0.1,
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

    cuDNN is disabled for this forward pass because cuDNN's LSTM kernel
    cannot handle the very long sequences produced by full MAESTRO pieces
    (CUDNN_STATUS_NOT_SUPPORTED). The native PyTorch LSTM kernel works
    at any sequence length with negligible speed difference for eval.

    Returns:
        Dict with keys: all metrics from compute_metrics(), error_analysis,
                        pred tensors for downstream use (prefixed with _),
                        piece_duration_sec, n_frames
    """
    data = load_from_cache(cache_path)
    mel  = data["mel"]          # (229, T_full)

    # Ground-truth rolls
    gt_onset    = data["onset"]     # (T_full, 88)
    gt_frame    = data["frame"]
    gt_offset   = data["offset"]
    gt_velocity = data["velocity"]

    T_full = mel.shape[1]

    # --- Full-length single-pass inference (Magenta + jongwook approach) ---
    # cuDNN disabled: its LSTM kernel fails on very long sequences.
    # Native PyTorch LSTM has no sequence length limit.
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        w_mel = mel.unsqueeze(0).to(device)
        out   = model(w_mel)

        pred_onset    = out["onset"][0].cpu()      # (T_full, 88)
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
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
        fps=FRAMES_PER_SECOND,
    )

    # Error analysis (project-specific supplementary metrics)
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

    # Piece metadata
    metrics["n_frames"]           = T_full
    metrics["piece_duration_sec"] = T_full / FRAMES_PER_SECOND

    # Attach predictions for optional downstream use
    metrics["_pred_onset"]    = pred_onset
    metrics["_pred_frame"]    = pred_frame
    metrics["_pred_offset"]   = pred_offset
    metrics["_pred_velocity"] = pred_velocity
    metrics["_gt_frame"]      = gt_frame
    metrics["_gt_onset"]      = gt_onset

    return metrics


# ---------------------------------------------------------------------------
# GPU info helper
# ---------------------------------------------------------------------------

def _get_gpu_info() -> Dict[str, str]:
    info = {"device": "cpu"}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = f"{props.total_memory / 1e9:.1f}"
        info["cuda"] = torch.version.cuda or "N/A"
        info["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    info["pytorch"] = torch.__version__
    return info
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
    onset_tolerance:     float = 0.05,
    offset_ratio:        float = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance:  float = 0.1,
) -> Dict:
    """
    Full evaluation run. Returns summary metrics dict.

    Outputs are written into:
        <checkpoint_dir>/../eval_<split>/
    """
    eval_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = _get_gpu_info()
    print(f"Device: {gpu_info.get('device', device)}")

    # Load model
    model = OnsetsAndFrames(model_complexity=model_complexity)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    n_params = model.count_parameters()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {ckpt.get('epoch','?')} epochs, "
          f"val_loss={ckpt.get('val_loss',0):.4f}")
    print(f"  Model parameters: {n_params:,}")

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

    total_in_split = len(split_df)
    if max_files:
        split_df = split_df.head(max_files)

    print(f"\nEvaluating {len(split_df)}/{total_in_split} files from '{split}' split...")
    print(f"  Strategy: full-length single-pass inference")
    

    all_metrics: List[Dict] = []
    per_file:    List[Dict] = []

    skipped   = 0
    eval_counter = 0

    for i, row in split_df.iterrows():
        audio_path = str(Path(maestro_root) / row["audio_filename"])
        stem       = Path(audio_path).stem
        cp         = _cache_path(audio_path, cache_dir)

        if not cp.exists():
            skipped += 1
            continue

        file_start = time.time()

        file_metrics = evaluate_file(
            model, cp, device,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
            velocity_tolerance=velocity_tolerance
        )

        file_elapsed = time.time() - file_start
        eval_counter += 1

        # Collect scalar metrics for averaging
        scalars = {k: v for k, v in file_metrics.items()
                   if not k.startswith("_") and k != "error_analysis"}
        # Flatten error_analysis into ea_ prefixed keys
        ea = file_metrics.get("error_analysis", {})
        scalars.update({f"ea_{k}": v for k, v in ea.items()})
        scalars["stem"] = stem
        scalars["eval_time_sec"] = round(file_elapsed, 2)
        per_file.append(scalars)
        all_metrics.append(scalars)

        # Progress (every 10 files)
        # if eval_counter % 10 == 0:
        #     print(f"  [{eval_counter}/{len(split_df) - skipped}] "
        #           f"{stem[:40]}... {file_elapsed:.1f}s")

        # Optional: save plots (use eval_counter, not DataFrame index i)
        if save_plots and eval_counter <= 10:
            plot_piano_roll_comparison(
                pred_frame=file_metrics["_pred_frame"],
                gt_frame=file_metrics["_gt_frame"],
                title=stem,
                save_path=plots_dir / f"{stem}.png",
            )

        # Optional: save MIDI
        if save_midi and eval_counter <= 20:
            pm = rolls_to_midi(
                onset_roll=file_metrics["_pred_onset"],
                frame_roll=file_metrics["_pred_frame"],
                velocity_roll=file_metrics["_pred_velocity"],
                fps=FRAMES_PER_SECOND,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
            )
            pm.write(str(midi_dir / f"{stem}.mid"))

    # ---------------------------------------------------------------------------
    # Compute summary
    # ---------------------------------------------------------------------------
    eval_elapsed = time.time() - eval_start_time
    total_requested = len(split_df)
    total_evaluated = len(all_metrics)
    print(f"\n  Evaluated {total_evaluated}/{total_requested} files"
          + (f" ({skipped} skipped — no cache)" if skipped > 0 else "")
          + f" in {eval_elapsed:.1f}s")

    if not all_metrics:
        print("No files evaluated.")
        return {}

    # ---------------------------------------------------------------------------
    # Build summary dict
    # ---------------------------------------------------------------------------
    summary = {}

    # Mean of all numeric per-file metrics
    numeric_keys = [k for k in all_metrics[0]
                    if k not in ("stem", "eval_time_sec")
                    and isinstance(all_metrics[0][k], (int, float))]
    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        summary[k] = float(np.mean(vals)) if vals else 0.0

    # --- Metadata for reproducibility ---
    summary["n_files"]           = total_evaluated
    summary["n_files_in_split"]  = total_in_split
    summary["split"]             = split
    summary["checkpoint"]        = str(checkpoint_path)
    summary["model_complexity"]  = model_complexity
    summary["model_parameters"]  = n_params
    summary["onset_threshold"]   = onset_threshold
    summary["frame_threshold"]   = frame_threshold
    summary["eval_time_total_s"] = round(eval_elapsed, 1)
    summary["eval_strategy"]     = "full_length_single_pass"

    # Dataset info
    summary["dataset"]           = "MAESTRO"
    summary["dataset_version"]   = "v3.0.0"
    summary["maestro_root"]      = str(maestro_root)

    # Evaluation protocol (locked tolerances)
    summary["eval_protocol"]     = get_eval_protocol(
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
    )

    # GPU / environment
    summary["gpu_info"]          = gpu_info

    # Training info from checkpoint
    summary["train_epochs"]      = ckpt.get("epoch", None)
    summary["train_val_loss"]    = ckpt.get("val_loss", None)
    summary["train_best_val_loss"] = ckpt.get("best_val_loss", None)

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    with open(eval_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_dir / "per_file_metrics.json", "w") as f:
        json.dump(per_file, f, indent=2)


        # Protocol reminder
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"{'— Decoding + evaluation protocol —':^50}")
    print(f"{'='*60}")
    print('\n')
    proto = summary["eval_protocol"]

    print("  Decode thresholds:")
    print(f"    onset_threshold:   {summary['onset_threshold']:.2f}")
    print(f"    frame_threshold:   {summary['frame_threshold']:.2f}")

    print("  Evaluation tolerances:")
    print(f"    onset_tolerance:   {proto['onset_tolerance_s']*1000:.0f} ms")
    print(f"    pitch_tolerance:   {proto['pitch_tolerance_raw']:.2f} "
        f"({proto['pitch_tolerance_cents']:.0f} cents)")
    print(f"    offset_ratio:      {proto['offset_ratio']}")
    print(f"    offset_min_tol:    {proto['offset_min_tolerance_s']*1000:.0f} ms")
    print(f"    velocity_tolerance:{proto['velocity_tolerance']}")
    print(f"    mir_eval version:  {proto['mir_eval_version']}")
    
    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — {split} split (n={total_evaluated})")
    print(f"{'='*60}")
    print('\n')
    print(f"  Dataset:    MAESTRO v3.0.0, {split} split")
    print(f"  Model:      OnsetsAndFrames (complexity={model_complexity}, "
          f"{n_params:,} params)")
    print(f"  Checkpoint: epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss',0):.4f}")
    print(f"  GPU:        {gpu_info.get('device', 'cpu')}")
    print(f"  Eval time:  {eval_elapsed:.1f}s "
          f"({eval_elapsed/total_evaluated:.1f}s/file)")
    print()

    # Primary metrics (paper-comparable)
    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"  {'— Primary metrics —':^50}")
    print(f"{'='*60}")
    print('\n')
    print(f"  {'Metric':<35s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}")
    print(f"  {'-'*60}")
    for prefix, label in [
        ("note",                "Note (onset+pitch)"),
        ("note_with_offset",    "Note w/ offset"),
        ("note_with_offset_vel","Note w/ offset+vel"),
        ("frame",               "Frame"),
    ]:
        p = summary.get(f"{prefix}_precision", 0)
        r = summary.get(f"{prefix}_recall", 0)
        f = summary.get(f"{prefix}_f1", 0)
        print(f"  {label:<35s}  {p:>7.4f}  {r:>7.4f}  {f:>7.4f}")

    print()

    # Note counts (diagnostic)
    n_pred = summary.get("n_pred_notes", 0)
    n_gt   = summary.get("n_gt_notes", 0)
    print(f"  Avg notes/file:  pred={n_pred:.0f}  gt={n_gt:.0f}  "
          f"ratio={n_pred/n_gt:.2f}" if n_gt > 0 else "")

    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"{'— Supplementary error analysis —':^50}")
    print(f"{'='*60}")
    print('\n')
    for key, label, fmt in [
        ("ea_offset_mae_ms",       "Offset MAE",           "{:.1f} ms"),
        ("ea_onset_mae_ms",        "Onset MAE",            "{:.1f} ms"),
        ("ea_chord_completeness",  "Chord completeness",   "{:.4f}"),
        ("ea_duplicate_note_rate", "Duplicate note rate",   "{:.4f}"),
    ]:
        val = summary.get(key, 0)
        print(f"  {label:<35s}  {fmt.format(val)}")


    print(f"\n  Results saved → {eval_dir}")
    print(f"{'='*60}\n")



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
