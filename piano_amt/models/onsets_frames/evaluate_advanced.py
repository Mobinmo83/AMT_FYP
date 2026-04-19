"""
models/onsets_frames/evaluate_advanced.py — Advanced evaluation harness.

Identical to evaluate.py but uses decode_advanced.py for post-processing.
This allows running the original and advanced evaluations side by side
from different notebooks without modifying any original code.

Key difference from evaluate.py:
  - Uses advanced_rolls_to_note_events() instead of rolls_to_note_events()
  - Accepts all post-processing toggles as parameters
  - Saves results to eval_<split>_<config_name>/ directories

Usage:
    python -m models.onsets_frames.evaluate_advanced \\
        --checkpoint  /path/to/best.pt \\
        --maestro_root /path/to/maestro-v3.0.0 \\
        --cache_dir    /path/to/cache \\
        --split test \\
        --config_name "pp_method1_3" \\
        --use_onset_conditioned_offset \\
        --min_note_duration_ms 50

    Or from a notebook:
        from models.onsets_frames.evaluate_advanced import run_advanced_evaluation

        summary = run_advanced_evaluation(
            checkpoint_path=best_ckpt,
            maestro_root=MAESTRO_ROOT,
            cache_dir=CACHE_DIR,
            split='test',
            config_name='pp_all',
            model_complexity=48,
            use_onset_conditioned_offset=True,
            use_frame_smoothing=True,
            min_note_duration_ms=50.0,
            use_duplicate_removal=True,
            use_chord_grouping=True,
        )
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
from models.onsets_frames.decode_advanced import advanced_rolls_to_note_events
from models.onsets_frames.decode import NoteEvent
from src.constants import N_KEYS, N_MELS, FRAMES_PER_SECOND, MAX_SEGMENT_FRAMES
from src.dataset import load_from_cache, _cache_path
from evaluate.metrics import compute_metrics, get_eval_protocol
# from evaluate.error_analysis import compute_error_analysis
from evaluate.plots import plot_piano_roll_comparison
from evaluate.metrics import compute_frame_metrics


# ---------------------------------------------------------------------------
# Event-level error analysis helper
# ---------------------------------------------------------------------------

def _compute_event_error_analysis(
    pred_events,
    gt_events,
    onset_tolerance_sec: float = 0.05,
    chord_window_sec: float = 0.05,
    duplicate_window_sec: float = 0.05,
) -> Dict:
    """
    Compute supplementary error-analysis metrics from decoded NoteEvent lists.
    Replaces the raw-roll compute_error_analysis() call so that offset MAE,
    chord completeness, and duplicate rate all reflect the post-processing config.
    """
    result = {
        "offset_mae_ms": 0.0,
        "onset_mae_ms": 0.0,
        "chord_completeness": 0.0,
        "duplicate_note_rate": 0.0,
    }

    if not pred_events or not gt_events:
        return result

    from collections import defaultdict

    # Build pitch index over pred events for fast lookup
    pred_by_pitch = defaultdict(list)
    for idx, e in enumerate(pred_events):
        pred_by_pitch[e.pitch].append(idx)

    pred_used = [False] * len(pred_events)
    offset_errs, onset_errs = [], []

    # Greedy match: each GT event finds closest unmatched pred event at same pitch
    for ge in gt_events:
        candidates = pred_by_pitch.get(ge.pitch, [])
        best_idx, best_dt = None, float("inf")
        for ci in candidates:
            if pred_used[ci]:
                continue
            dt = abs(pred_events[ci].onset_sec - ge.onset_sec)
            if dt < best_dt:
                best_dt, best_idx = dt, ci
        if best_idx is not None and best_dt <= onset_tolerance_sec:
            pred_used[best_idx] = True
            pe = pred_events[best_idx]
            onset_errs.append(abs(pe.onset_sec - ge.onset_sec) * 1000)
            offset_errs.append(abs(pe.offset_sec - ge.offset_sec) * 1000)

    if onset_errs:
        result["onset_mae_ms"] = float(np.mean(onset_errs))
    if offset_errs:
        result["offset_mae_ms"] = float(np.mean(offset_errs))

    # Chord completeness: for each GT chord (>=2 simultaneous notes),
    # what fraction of tones appear in pred?
    chord_scores = []
    gt_sorted = sorted(gt_events, key=lambda e: e.onset_sec)
    i = 0
    while i < len(gt_sorted):
        chord = [gt_sorted[i]]
        j = i + 1
        while j < len(gt_sorted) and (gt_sorted[j].onset_sec - gt_sorted[i].onset_sec) <= chord_window_sec:
            chord.append(gt_sorted[j])
            j += 1
        if len(chord) >= 2:
            found = sum(
                1 for tone in chord
                if any(
                    pe.pitch == tone.pitch and abs(pe.onset_sec - tone.onset_sec) <= onset_tolerance_sec
                    for pe in pred_events
                )
            )
            chord_scores.append(found / len(chord))
        i = j

    if chord_scores:
        result["chord_completeness"] = float(np.mean(chord_scores))

    # Duplicate note rate: pred events with same pitch within window of each other
    n_dups = 0
    pred_sorted = sorted(pred_events, key=lambda e: (e.pitch, e.onset_sec))
    for k in range(1, len(pred_sorted)):
        prev, curr = pred_sorted[k - 1], pred_sorted[k]
        if curr.pitch == prev.pitch and (curr.onset_sec - prev.onset_sec) <= duplicate_window_sec:
            n_dups += 1
    result["duplicate_note_rate"] = n_dups / len(pred_events) if pred_events else 0.0

    return result


# ---------------------------------------------------------------------------
# Per-file evaluation (advanced version)
# ---------------------------------------------------------------------------



def evaluate_file_advanced(
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
    # Post-processing kwargs passed to advanced decoder
    **pp_kwargs,
) -> Dict:
    """
    Run model on one full-length cached file and compute all metrics
    using the advanced decoder with post-processing.

    Same evaluation strategy as evaluate.py (full-length single-pass),
    but note events are decoded via advanced_rolls_to_note_events().

    Returns:
        Dict with all metrics, error analysis, and prediction tensors.
    """
    data = load_from_cache(cache_path)
    mel = data["mel"]           # (229, T_full)
    gt_onset = data["onset"]    # (T_full, 88)
    gt_frame = data["frame"]
    gt_offset = data["offset"]
    gt_velocity = data["velocity"]

    T_full = mel.shape[1]

    # Full-length single-pass inference (same as evaluate.py)
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        w_mel = mel.unsqueeze(0).to(device)
        out = model(w_mel)
        pred_onset = out["onset"][0].cpu()
        pred_frame = out["frame"][0].cpu()
        pred_offset = out["offset"][0].cpu()
        pred_velocity = out["velocity"][0].cpu()

    # Standard metrics using original decode (for fair comparison baseline)
    metrics = compute_frame_metrics(
        pred_frame=pred_frame,
        gt_frame=gt_frame,
        frame_threshold=frame_threshold,
    )

    # ---- Advanced: decode with post-processing ----
    # Decode predicted events using advanced method
    pred_events_advanced = advanced_rolls_to_note_events(
        onset_roll=pred_onset,
        frame_roll=pred_frame,
        offset_roll=pred_offset,
        velocity_roll=pred_velocity,
        fps=FRAMES_PER_SECOND,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        offset_threshold=offset_threshold,
        **pp_kwargs,
    )

    # Decode GT events (using standard decode, no post-processing)
    from models.onsets_frames.decode import rolls_to_note_events
    gt_events = rolls_to_note_events(
        onset_roll=gt_onset,
        frame_roll=gt_frame,
        velocity_roll=gt_velocity,
        fps=FRAMES_PER_SECOND,
        onset_threshold=0.5,
        frame_threshold=0.5,
    )

    # Compute advanced note-level metrics via mir_eval
    try:
        import mir_eval
        from mir_eval.transcription_velocity import precision_recall_f1_overlap as eval_vel

        # Convert events to mir_eval format
        def events_to_mir(events):
            if not events:
                return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
            intervals = np.array([[e.onset_sec, e.offset_sec] for e in events])
            pitches = np.array([float(e.pitch) for e in events])
            velocities = np.array([float(e.velocity) for e in events])
            return intervals, pitches, velocities

        pred_int, pred_pit, pred_vel = events_to_mir(pred_events_advanced)
        gt_int, gt_pit, gt_vel = events_to_mir(gt_events)

        adv_metrics = {}
        adv_metrics["adv_n_pred_notes"] = len(pred_pit)   # keep for trace
        adv_metrics["adv_n_gt_notes"] = len(gt_pit)        # keep for trace
        adv_metrics["n_pred_notes"] = len(pred_pit)        # overwrite standard key
        adv_metrics["n_gt_notes"] = len(gt_pit)            # overwrite standard key

        if len(pred_pit) > 0 and len(gt_pit) > 0:
            # Tier 1: Note F1
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=gt_int, ref_pitches=gt_pit,
                est_intervals=pred_int, est_pitches=pred_pit,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=0.25,
                offset_ratio=None, offset_min_tolerance=None,
            )
            adv_metrics["adv_note_precision"] = float(p)
            adv_metrics["note_precision"] = float(p)           # overwrite standard key

            adv_metrics["adv_note_recall"] = float(r)
            adv_metrics["note_recall"] = float(r)

            adv_metrics["adv_note_f1"] = float(f)
            adv_metrics["note_f1"] = float(f)

            # Tier 2: Note+Offset F1
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=gt_int, ref_pitches=gt_pit,
                est_intervals=pred_int, est_pitches=pred_pit,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=0.25,
                offset_ratio=offset_ratio,
                offset_min_tolerance=offset_min_tolerance,
            )
            adv_metrics["adv_note_with_offset_precision"] = float(p)
            adv_metrics["note_with_offset_precision"] = float(p)
            adv_metrics["adv_note_with_offset_recall"] = float(r)
            adv_metrics["note_with_offset_recall"] = float(r)
            adv_metrics["adv_note_with_offset_f1"] = float(f)
            adv_metrics["note_with_offset_f1"] = float(f)
            # Tier 3: Note+Offset+Velocity F1
            p, r, f, _ = eval_vel(
                ref_intervals=gt_int, ref_pitches=gt_pit, ref_velocities=gt_vel,
                est_intervals=pred_int, est_pitches=pred_pit, est_velocities=pred_vel,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=0.25,
                offset_ratio=offset_ratio,
                offset_min_tolerance=offset_min_tolerance,
                velocity_tolerance=velocity_tolerance,
            )
            adv_metrics["adv_note_with_offset_vel_precision"] = float(p)
            adv_metrics["note_with_offset_vel_precision"] = float(p)
            adv_metrics["adv_note_with_offset_vel_recall"] = float(r)
            adv_metrics["note_with_offset_vel_recall"] = float(r)
            adv_metrics["adv_note_with_offset_vel_f1"] = float(f)
            adv_metrics["note_with_offset_vel_f1"] = float(f)
        else:
            for k in ["adv_note_precision", "adv_note_recall", "adv_note_f1",
                       "adv_note_with_offset_precision", "adv_note_with_offset_recall",
                       "adv_note_with_offset_f1", "adv_note_with_offset_vel_precision",
                       "adv_note_with_offset_vel_recall", "adv_note_with_offset_vel_f1"]:
                adv_metrics[k] = 0.0
            for k in ["note_precision", "note_recall", "note_f1",
                       "note_with_offset_precision", "note_with_offset_recall",
                       "note_with_offset_f1", "note_with_offset_vel_precision",
                       "note_with_offset_vel_recall", "note_with_offset_vel_f1",
                       "n_pred_notes", "n_gt_notes"]:
                adv_metrics[k] = 0.0

        metrics.update(adv_metrics)

    except ImportError:
        print("WARNING: mir_eval not installed — advanced note metrics skipped.")

    # Error analysis — event-level, reflects the chosen post-processing config.
    # Uses pred_events_advanced (already decoded above), not raw rolls.
    ea = _compute_event_error_analysis(
        pred_events=pred_events_advanced,
        gt_events=gt_events,
    )
    metrics["error_analysis"] = ea

    # Piece metadata
    metrics["n_frames"] = T_full
    metrics["piece_duration_sec"] = T_full / FRAMES_PER_SECOND

    # Attach predictions for downstream use
    metrics["_pred_onset"] = pred_onset
    metrics["_pred_frame"] = pred_frame
    metrics["_pred_offset"] = pred_offset
    metrics["_pred_velocity"] = pred_velocity
    metrics["_gt_frame"] = gt_frame
    metrics["_gt_onset"] = gt_onset

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


def _pick_metric(summary: Dict, adv_key: str, base_key: str) -> float:
    val = summary.get(adv_key, None)
    if val is not None:
        return float(val)
    return float(summary.get(base_key, 0.0))


def _on_off(flag: bool) -> str:
    return "ON" if flag else "OFF"


# ---------------------------------------------------------------------------
# Main advanced evaluation loop
# ---------------------------------------------------------------------------

def run_advanced_evaluation(
    checkpoint_path: str | Path,
    maestro_root: str | Path,
    cache_dir: str | Path,
    split: str = "test",
    config_name: str = "advanced",
    max_files: Optional[int] = None,
    save_midi: bool = False,
    save_plots: bool = False,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    model_complexity: int = 48,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance: float = 0.1,
    # Post-processing toggles
    use_onset_conditioned_offset: bool = False,
    use_frame_smoothing: bool = False,
    frame_smoothing_kernel: int = 7,
    frame_smoothing_method: str = "median",
    min_note_duration_ms: float = 16.0,
    use_duplicate_removal: bool = False,
    duplicate_tolerance_sec: float = 0.05,
    use_chord_grouping: bool = False,
    chord_tolerance_sec: float = 0.03,
    chord_snap_to: str = "median",
    use_adaptive_thresholds: bool = False,
    adaptive_onset_k: float = 0.5,
    adaptive_frame_k: float = 0.5,
    use_pedal_extension: bool = False,
    pedal_energy_threshold: float = 10.0,
    pedal_max_extension_sec: float = 2.0,
) -> Dict:
    """
    Full advanced evaluation run. Returns summary metrics dict.

    Results saved to: <run_dir>/eval_<split>_<config_name>/
    """
    eval_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = _get_gpu_info()
    print(f"Device: {gpu_info.get('device', device)}")

    # Load model
    model = OnsetsAndFrames(model_complexity=model_complexity)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    n_params = model.count_parameters()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs, "
          f"val_loss={ckpt.get('val_loss', 0):.4f}")
    print(f"  Model parameters: {n_params:,}")

    # Build post-processing kwargs dict
    pp_kwargs = {
        "use_onset_conditioned_offset": use_onset_conditioned_offset,
        "use_frame_smoothing": use_frame_smoothing,
        "frame_smoothing_kernel": frame_smoothing_kernel,
        "frame_smoothing_method": frame_smoothing_method,
        "min_note_duration_ms": min_note_duration_ms,
        "use_duplicate_removal": use_duplicate_removal,
        "duplicate_tolerance_sec": duplicate_tolerance_sec,
        "use_chord_grouping": use_chord_grouping,
        "chord_tolerance_sec": chord_tolerance_sec,
        "chord_snap_to": chord_snap_to,
        "use_adaptive_thresholds": use_adaptive_thresholds,
        "adaptive_onset_k": adaptive_onset_k,
        "adaptive_frame_k": adaptive_frame_k,
        "use_pedal_extension": use_pedal_extension,
        "pedal_energy_threshold": pedal_energy_threshold,
        "pedal_max_extension_sec": pedal_max_extension_sec,
    }

    # Print active post-processing methods
    print(f"\n  Post-processing config: '{config_name}'")
    active_methods = []
    if use_frame_smoothing:
        active_methods.append(f"Frame Smoothing (kernel={frame_smoothing_kernel}, method={frame_smoothing_method})")
    if use_onset_conditioned_offset:
        active_methods.append("Onset-Conditioned Offset")
    if min_note_duration_ms != 16.0:
        active_methods.append(f"Min Note Duration = {min_note_duration_ms}ms")
    if use_duplicate_removal:
        active_methods.append(f"Duplicate Removal (tol={duplicate_tolerance_sec*1000:.0f}ms)")
    if use_chord_grouping:
        active_methods.append(f"Chord Grouping (tol={chord_tolerance_sec*1000:.0f}ms, snap={chord_snap_to})")
    if use_adaptive_thresholds:
        active_methods.append(f"Adaptive Thresholds (onset_k={adaptive_onset_k}, frame_k={adaptive_frame_k})")
    if use_pedal_extension:
        active_methods.append(f"Pedal Extension (threshold={pedal_energy_threshold}, max={pedal_max_extension_sec}s)")

    if active_methods:
        for m in active_methods:
            print(f"    ✓ {m}")
    else:
        print("    (no post-processing — baseline decode)")

    # Output directory
    ckpt_dir = Path(checkpoint_path).parent
    run_dir = ckpt_dir.parent
    eval_dir = run_dir / f"eval_{split}_{config_name}"
    eval_dir.mkdir(exist_ok=True)
    plots_dir = eval_dir / "plots"
    midi_dir = eval_dir / "midi_samples"
    if save_plots:
        plots_dir.mkdir(exist_ok=True)
    if save_midi:
        midi_dir.mkdir(exist_ok=True)

    # Collect files
    import pandas as pd
    csv_files = sorted(Path(maestro_root).glob("*.csv"))
    df = pd.read_csv(csv_files[0])
    split_df = df[df["split"] == split].reset_index(drop=True)
    total_in_split = len(split_df)
    if max_files:
        split_df = split_df.head(max_files)

    print(f"\nEvaluating {len(split_df)}/{total_in_split} files from '{split}' split...")
    print(f"  Strategy: full-length single-pass inference + advanced post-processing")

    all_metrics: List[Dict] = []
    per_file: List[Dict] = []
    skipped = 0
    eval_counter = 0

    for i, row in split_df.iterrows():
        audio_path = str(Path(maestro_root) / row["audio_filename"])
        stem = Path(audio_path).stem
        cp = _cache_path(audio_path, cache_dir)

        if not cp.exists():
            skipped += 1
            continue

        file_start = time.time()

        file_metrics = evaluate_file_advanced(
            model, cp, device,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            onset_tolerance=onset_tolerance,
            offset_threshold=offset_threshold,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
            velocity_tolerance=velocity_tolerance,
            **pp_kwargs,
        )

        file_elapsed = time.time() - file_start
        eval_counter += 1

        # Collect scalar metrics
        scalars = {k: v for k, v in file_metrics.items()
                   if not k.startswith("_") and k != "error_analysis"}
        ea = file_metrics.get("error_analysis", {})
        scalars.update({f"ea_{k}": v for k, v in ea.items()})
        scalars["stem"] = stem
        scalars["eval_time_sec"] = round(file_elapsed, 2)
        per_file.append(scalars)
        all_metrics.append(scalars)

        # Optional plots
        if save_plots and eval_counter <= 10:
            plot_piano_roll_comparison(
                pred_frame=file_metrics["_pred_frame"],
                gt_frame=file_metrics["_gt_frame"],
                title=f"{stem} ({config_name})",
                save_path=plots_dir / f"{stem}.png",
            )

        # Optional MIDI (using advanced decoder)
        if save_midi and eval_counter <= 20:
            from models.onsets_frames.decode_advanced import advanced_rolls_to_midi_file
            advanced_rolls_to_midi_file(
                onset_roll=file_metrics["_pred_onset"],
                frame_roll=file_metrics["_pred_frame"],
                offset_roll=file_metrics["_pred_offset"],
                velocity_roll=file_metrics["_pred_velocity"],
                output_path=midi_dir / f"{stem}.mid",
                fps=FRAMES_PER_SECOND,
                onset_threshold=onset_threshold,
                offset_threshold=offset_threshold,
                frame_threshold=frame_threshold,
                **pp_kwargs,
            )

    # Compute summary
    eval_elapsed = time.time() - eval_start_time
    total_requested = len(split_df)
    total_evaluated = len(all_metrics)
    print(f"\n  Evaluated {total_evaluated}/{total_requested} files"
          + (f" ({skipped} skipped — no cache)" if skipped > 0 else "")
          + f" in {eval_elapsed:.1f}s")

    if not all_metrics:
        print("No files evaluated.")
        return {}

    summary = {}
    numeric_keys = [k for k in all_metrics[0]
                    if k not in ("stem", "eval_time_sec")
                    and isinstance(all_metrics[0][k], (int, float))]
    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        summary[k] = float(np.mean(vals)) if vals else 0.0

    # Metadata
     # Metadata
    summary["n_files"] = total_evaluated
    summary["n_files_in_split"] = total_in_split
    summary["split"] = split
    summary["config_name"] = config_name
    summary["checkpoint"] = str(checkpoint_path)
    summary["model_complexity"] = model_complexity
    summary["model_parameters"] = n_params
    summary["onset_threshold"] = onset_threshold
    summary["frame_threshold"] = frame_threshold
    summary["offset_threshold"] = offset_threshold
    summary["eval_time_total_s"] = round(eval_elapsed, 1)
    summary["eval_strategy"] = "full_length_single_pass_advanced"

    summary["dataset"] = "MAESTRO"
    summary["dataset_version"] = "v3.0.0"
    summary["maestro_root"] = str(maestro_root)

    summary["post_processing"] = pp_kwargs
    summary["eval_protocol"] = get_eval_protocol(
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
    )
    summary["gpu_info"] = gpu_info

    summary["train_epochs"] = ckpt.get("epoch", None)
    summary["train_val_loss"] = ckpt.get("val_loss", None)
    summary["train_best_val_loss"] = ckpt.get("best_val_loss", None)

    # Helpful trace fields
    summary["note_metric_source"] = "advanced_decoder"
    summary["frame_metric_source"] = "standard_frame_metric_path"
    

    # Save
    with open(eval_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_dir / "per_file_metrics.json", "w") as f:
        json.dump(per_file, f, indent=2)

    # Print results
    # ---------------------------------------------------------------------------
    # Print results (same style as original evaluator, but using advanced note metrics)
    # ---------------------------------------------------------------------------

    adv_note_p = _pick_metric(summary, "adv_note_precision", "note_precision")
    adv_note_r = _pick_metric(summary, "adv_note_recall", "note_recall")
    adv_note_f = _pick_metric(summary, "adv_note_f1", "note_f1")

    adv_off_p = _pick_metric(summary, "adv_note_with_offset_precision", "note_with_offset_precision")
    adv_off_r = _pick_metric(summary, "adv_note_with_offset_recall", "note_with_offset_recall")
    adv_off_f = _pick_metric(summary, "adv_note_with_offset_f1", "note_with_offset_f1")

    adv_vel_p = _pick_metric(summary, "adv_note_with_offset_vel_precision", "note_with_offset_vel_precision")
    adv_vel_r = _pick_metric(summary, "adv_note_with_offset_vel_recall", "note_with_offset_vel_recall")
    adv_vel_f = _pick_metric(summary, "adv_note_with_offset_vel_f1", "note_with_offset_vel_f1")

    adv_n_pred = _pick_metric(summary, "adv_n_pred_notes", "n_pred_notes")
    adv_n_gt = _pick_metric(summary, "adv_n_gt_notes", "n_gt_notes")

    print('\n')
    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"{'— Decoding + evaluation protocol —':^50}")
    print(f"{'='*60}")
    print()
    proto = summary["eval_protocol"]

    print("  Decode thresholds:")
    print(f"    onset_threshold:   {summary['onset_threshold']:.2f}")
    print(f"    frame_threshold:   {summary['frame_threshold']:.2f}")
    print(f"    offset_threshold:  {summary['offset_threshold']:.2f}")

    print("  Evaluation tolerances:")
    print(f"    onset_tolerance:   {proto['onset_tolerance_s']*1000:.0f} ms")
    print(f"    pitch_tolerance:   {proto['pitch_tolerance_raw']:.2f} "
          f"({proto['pitch_tolerance_cents']:.0f} cents)")
    print(f"    offset_ratio:      {proto['offset_ratio']}")
    print(f"    offset_min_tol:    {proto['offset_min_tolerance_s']*1000:.0f} ms")
    print(f"    velocity_tolerance:{proto['velocity_tolerance']}")
    print(f"    mir_eval version:  {proto['mir_eval_version']}")

    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — {split} split (n={total_evaluated})")
    print(f"{'='*60}")
    print()

    print(f"  Dataset:    MAESTRO v3.0.0, {split} split")
    print(f"  Model:      OnsetsAndFrames (complexity={model_complexity}, {n_params:,} params)")
    print(f"  Checkpoint: epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.4f}")
    print(f"  Config:     {config_name}")
    print(f"  GPU:        {gpu_info.get('device', 'cpu')}")
    print(f"  Eval time:  {eval_elapsed:.1f}s ({eval_elapsed/total_evaluated:.1f}s/file)")
    print()

    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"  {'— Primary metrics —':^50}")
    print(f"{'='*60}")
    print()

    print(f"  {'Metric':<35s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}")
    print(f"  {'-'*60}")

    print(f"  {'Note (onset+pitch)':<35s}  {adv_note_p:>7.4f}  {adv_note_r:>7.4f}  {adv_note_f:>7.4f}")
    print(f"  {'Note w/ offset':<35s}  {adv_off_p:>7.4f}  {adv_off_r:>7.4f}  {adv_off_f:>7.4f}")
    print(f"  {'Note w/ offset+vel':<35s}  {adv_vel_p:>7.4f}  {adv_vel_r:>7.4f}  {adv_vel_f:>7.4f}")
    print(f"  {'Frame':<35s}  {summary.get('frame_precision', 0):>7.4f}  "
        f"{summary.get('frame_recall', 0):>7.4f}  {summary.get('frame_f1', 0):>7.4f}")
    print('\n')
    print(f"  {'-'*60}")

    if adv_n_gt > 0:
        print(f"  Avg notes/file:  pred={adv_n_pred:.0f}  gt={adv_n_gt:.0f}  ratio={adv_n_pred/adv_n_gt:.2f}")

    print('\n')
    print('\n')
    print(f"\n{'='*60}")
    print(f"{'— Supplementary error analysis —':^50}")
    print(f"{'='*60}")
    print()

    for key, label, fmt in [
        ("ea_offset_mae_ms",       "Offset MAE",            "{:.1f} ms"),
        ("ea_onset_mae_ms",        "Onset MAE",             "{:.1f} ms"),
        ("ea_chord_completeness",  "Chord completeness",    "{:.4f}"),
        ("ea_duplicate_note_rate", "Duplicate note rate",   "{:.4f}"),
    ]:
        val = summary.get(key, 0)
        print(f"  {label:<35s}  {fmt.format(val)}")

    print(f"\n  Results saved → {eval_dir}")
    print(f"{'='*60}\n")

    return summary
