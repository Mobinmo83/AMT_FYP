from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import mir_eval
from mir_eval.transcription_velocity import precision_recall_f1_overlap as eval_vel
import numpy as np
import pandas as pd
import torch

from demo.decoder_presets import AdvancedDecoderConfig, DEFAULT_MODE, get_decoder_preset, make_decoder_config
from demo.inference import DemoNoteEvent, gt_rolls_to_note_events, prediction_to_note_events
from evaluate.metrics import compute_frame_metrics, get_eval_protocol
from models.onsets_frames.decode_advanced import compute_adaptive_thresholds, smooth_frame_roll
from src.constants import FRAMES_PER_SECOND

GroundTruthDict = Dict[str, torch.Tensor]
PredictionDict = Dict[str, torch.Tensor]


def _events_to_mir(events: Iterable[DemoNoteEvent]):
    events = list(events)
    if not events:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    intervals = np.array([[e.onset_sec, e.offset_sec] for e in events], dtype=float)
    pitches = np.array([float(e.pitch) for e in events], dtype=float)
    velocities = np.array([float(e.velocity) for e in events], dtype=float)
    return intervals, pitches, velocities


def _event_error_analysis(
    pred_events: list[DemoNoteEvent],
    gt_events: list[DemoNoteEvent],
    onset_tolerance_sec: float = 0.05,
    chord_window_sec: float = 0.05,
    duplicate_window_sec: float = 0.05,
) -> Dict[str, float]:
    result = {
        "offset_mae_ms": 0.0,
        "onset_mae_ms": 0.0,
        "chord_completeness": 0.0,
        "duplicate_note_rate": 0.0,
    }
    if not pred_events or not gt_events:
        return result

    pred_by_pitch = defaultdict(list)
    for idx, e in enumerate(pred_events):
        pred_by_pitch[e.pitch].append(idx)
    pred_used = [False] * len(pred_events)
    onset_errs, offset_errs = [], []
    for ge in gt_events:
        best_idx, best_dt = None, float("inf")
        for ci in pred_by_pitch.get(ge.pitch, []):
            if pred_used[ci]:
                continue
            dt = abs(pred_events[ci].onset_sec - ge.onset_sec)
            if dt < best_dt:
                best_dt, best_idx = dt, ci
        if best_idx is not None and best_dt <= onset_tolerance_sec:
            pred_used[best_idx] = True
            pe = pred_events[best_idx]
            onset_errs.append(abs(pe.onset_sec - ge.onset_sec) * 1000.0)
            offset_errs.append(abs(pe.offset_sec - ge.offset_sec) * 1000.0)
    if onset_errs:
        result["onset_mae_ms"] = float(np.mean(onset_errs))
    if offset_errs:
        result["offset_mae_ms"] = float(np.mean(offset_errs))

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
                if any(pe.pitch == tone.pitch and abs(pe.onset_sec - tone.onset_sec) <= onset_tolerance_sec for pe in pred_events)
            )
            chord_scores.append(found / len(chord))
        i = j
    if chord_scores:
        result["chord_completeness"] = float(np.mean(chord_scores))

    n_dups = 0
    pred_sorted = sorted(pred_events, key=lambda e: (e.pitch, e.onset_sec))
    for k in range(1, len(pred_sorted)):
        prev, curr = pred_sorted[k - 1], pred_sorted[k]
        if curr.pitch == prev.pitch and (curr.onset_sec - prev.onset_sec) <= duplicate_window_sec:
            n_dups += 1
    result["duplicate_note_rate"] = n_dups / len(pred_events) if pred_events else 0.0
    return result


def _frame_metrics_like_evaluate_advanced(
    pred: PredictionDict,
    gt: GroundTruthDict,
    cfg: AdvancedDecoderConfig,
) -> Dict[str, float]:
    effective_frame_threshold = cfg.frame_threshold
    frame_roll_for_metrics = pred["frame"]

    if cfg.use_adaptive_thresholds:
        _, effective_frame_threshold = compute_adaptive_thresholds(
            onset_roll=pred["onset"],
            frame_roll=pred["frame"],
            onset_base=cfg.onset_threshold,
            frame_base=cfg.frame_threshold,
            onset_k=cfg.adaptive_onset_k,
            frame_k=cfg.adaptive_frame_k,
        )
    if cfg.use_frame_smoothing:
        frame_roll_for_metrics = smooth_frame_roll(
            frame_roll=pred["frame"],
            kernel_size=cfg.frame_smoothing_kernel,
            method=cfg.frame_smoothing_method,
        )
    return compute_frame_metrics(
        pred_frame=frame_roll_for_metrics,
        gt_frame=gt["frame"],
        frame_threshold=effective_frame_threshold,
    )


def note_event_metrics(
    pred_events: list[DemoNoteEvent],
    gt_events: list[DemoNoteEvent],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance: float = 0.1,
) -> Dict[str, float]:
    pred_events = [e for e in pred_events if (e.offset_sec - e.onset_sec) >= 1.0 / FRAMES_PER_SECOND]
    pred_int, pred_pit, pred_vel = _events_to_mir(pred_events)
    gt_int, gt_pit, gt_vel = _events_to_mir(gt_events)
    metrics: Dict[str, float] = {
        "n_pred_notes": float(len(pred_pit)),
        "n_gt_notes": float(len(gt_pit)),
    }
    if len(pred_pit) == 0 or len(gt_pit) == 0:
        for k in [
            "note_precision", "note_recall", "note_f1",
            "note_with_offset_precision", "note_with_offset_recall", "note_with_offset_f1",
            "note_with_offset_vel_precision", "note_with_offset_vel_recall", "note_with_offset_vel_f1",
        ]:
            metrics[k] = 0.0
        return metrics

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=gt_int,
        ref_pitches=gt_pit,
        est_intervals=pred_int,
        est_pitches=pred_pit,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=0.25,
        offset_ratio=None,
        offset_min_tolerance=None,
    )
    metrics.update({"note_precision": float(p), "note_recall": float(r), "note_f1": float(f)})

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=gt_int,
        ref_pitches=gt_pit,
        est_intervals=pred_int,
        est_pitches=pred_pit,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=0.25,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )
    metrics.update({"note_with_offset_precision": float(p), "note_with_offset_recall": float(r), "note_with_offset_f1": float(f)})

    p, r, f, _ = eval_vel(
        ref_intervals=gt_int,
        ref_pitches=gt_pit,
        ref_velocities=gt_vel,
        est_intervals=pred_int,
        est_pitches=pred_pit,
        est_velocities=pred_vel,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=0.25,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
    )
    metrics.update({"note_with_offset_vel_precision": float(p), "note_with_offset_vel_recall": float(r), "note_with_offset_vel_f1": float(f)})
    return metrics


def _greedy_match_summary(pred_events: list[DemoNoteEvent], gt_events: list[DemoNoteEvent], tol_s: float = 0.05) -> Dict[str, float]:
    used_pred = set()
    matched_pairs = []
    for gt_e in gt_events:
        best_i = None
        best_key = (float("inf"), float("inf"))
        for i, pred_e in enumerate(pred_events):
            if i in used_pred or pred_e.pitch != gt_e.pitch:
                continue
            onset_err = abs(pred_e.onset_sec - gt_e.onset_sec)
            if onset_err > tol_s:
                continue
            offset_err = abs(pred_e.offset_sec - gt_e.offset_sec)
            key = (onset_err, offset_err)
            if key < best_key:
                best_key = key
                best_i = i
        if best_i is not None:
            used_pred.add(best_i)
            matched_pairs.append((gt_e, pred_events[best_i]))
    onset_errs_ms = [abs(pred_e.onset_sec - gt_e.onset_sec) * 1000.0 for gt_e, pred_e in matched_pairs]
    offset_errs_ms = [abs(pred_e.offset_sec - gt_e.offset_sec) * 1000.0 for gt_e, pred_e in matched_pairs]
    return {
        "matched_notes": len(matched_pairs),
        "missed_notes": len(gt_events) - len(matched_pairs),
        "extra_notes": len(pred_events) - len(matched_pairs),
        "predicted_notes": len(pred_events),
        "ground_truth_notes": len(gt_events),
        "onset_timing_mae_ms": float(np.mean(onset_errs_ms)) if onset_errs_ms else 0.0,
        "offset_timing_mae_ms": float(np.mean(offset_errs_ms)) if offset_errs_ms else 0.0,
    }


def evaluate_prediction_vs_gt(
    pred: PredictionDict,
    gt: GroundTruthDict,
    decoder_config: AdvancedDecoderConfig | None = None,
    decoder_mode: str = DEFAULT_MODE,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance: float = 0.1,
    **decoder_overrides,
) -> Dict[str, Dict[str, float]]:
    cfg = decoder_config or make_decoder_config(decoder_mode, **decoder_overrides)
    pred_events = prediction_to_note_events(pred, decoder_config=cfg)
    gt_events = gt_rolls_to_note_events(gt)
    metrics = note_event_metrics(
        pred_events,
        gt_events,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
    )
    metrics.update(_frame_metrics_like_evaluate_advanced(pred, gt, cfg))
    error_analysis = _event_error_analysis(pred_events, gt_events, onset_tolerance_sec=onset_tolerance)
    failure = _greedy_match_summary(pred_events, gt_events, tol_s=onset_tolerance)
    return {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "failure": failure,
        "protocol": get_eval_protocol(
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
            velocity_tolerance=velocity_tolerance,
        ),
        "decoder": {
            "mode": decoder_mode,
            "config_name": cfg.name,
            "label": cfg.label,
            "onset_threshold": cfg.onset_threshold,
            "frame_threshold": cfg.frame_threshold,
            "offset_threshold": cfg.offset_threshold,
            "gt_source": "cached_label_rolls_baseline_decode_0.5_0.5",
            "decoder_type": cfg.decoder_type,
            "advanced_kwargs": cfg.decoder_kwargs(),
        },
        "events": {
            "pred": pred_events,
            "gt_eval": gt_events,
        },
    }


def main_scores_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    m = results["metrics"]

    return pd.DataFrame([
        {
            "Evaluation metric": "Note",
            "Precision": m.get("note_precision", 0.0),
            "Recall": m.get("note_recall", 0.0),
            "F1-score": m.get("note_f1", 0.0),
        },
        {
            "Evaluation metric": "Note + offset",
            "Precision": m.get("note_with_offset_precision", 0.0),
            "Recall": m.get("note_with_offset_recall", 0.0),
            "F1-score": m.get("note_with_offset_f1", 0.0),
        },
        {
            "Evaluation metric": "Note + offset + velocity",
            "Precision": m.get("note_with_offset_vel_precision", 0.0),
            "Recall": m.get("note_with_offset_vel_recall", 0.0),
            "F1-score": m.get("note_with_offset_vel_f1", 0.0),
        },
        {
            "Evaluation metric": "Frame",
            "Precision": m.get("frame_precision", 0.0),
            "Recall": m.get("frame_recall", 0.0),
            "F1-score": m.get("frame_f1", 0.0),
        },
    ])


def style_main_scores_table(df: pd.DataFrame):
    """Clean notebook display for the four main transcription metrics."""
    return (
        df.style
        .hide(axis="index")
        .format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-score": "{:.4f}",
        })
        .set_properties(**{
            "text-align": "center",
            "font-size": "13px",
            "padding": "6px",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("font-size", "13px"),
                    ("padding", "6px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #ddd"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "8px 0"),
                ],
            },
        ])
    )


def supplementary_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    ea = results.get("error_analysis", {})
    f = results["failure"]
    return pd.DataFrame([
        {"Item": "Matched notes", "Value": f.get("matched_notes", 0)},
        {"Item": "Missed notes", "Value": f.get("missed_notes", 0)},
        {"Item": "Extra notes", "Value": f.get("extra_notes", 0)},
        {"Item": "Predicted notes", "Value": f.get("predicted_notes", 0)},
        {"Item": "Ground-truth notes", "Value": f.get("ground_truth_notes", 0)},
        {"Item": "Onset timing MAE (ms)", "Value": f.get("onset_timing_mae_ms", 0.0)},
        {"Item": "Offset timing MAE (ms)", "Value": f.get("offset_timing_mae_ms", 0.0)},
        {"Item": "Offset MAE (advanced error analysis, ms)", "Value": ea.get("offset_mae_ms", 0.0)},
        {"Item": "Onset MAE (advanced error analysis, ms)", "Value": ea.get("onset_mae_ms", 0.0)},
        {"Item": "Chord completeness", "Value": ea.get("chord_completeness", 0.0)},
        {"Item": "Duplicate note rate", "Value": ea.get("duplicate_note_rate", 0.0)},
    ])


def decoder_config_table(cfg: AdvancedDecoderConfig) -> pd.DataFrame:
    return pd.DataFrame([{"Parameter": k, "Value": v} for k, v in cfg.__dict__.items()])


def compare_decoder_modes(
    pred: PredictionDict,
    gt: GroundTruthDict,
    modes: list[str] | None = None,
    configs: list[AdvancedDecoderConfig] | None = None,
) -> pd.DataFrame:
    rows = []
    if configs is None:
        modes = modes or ["baseline", "efficient_m3_m4", "quality_m2_m3_m4"]
        configs = [get_decoder_preset(m) for m in modes]
    for cfg in configs:
        r = evaluate_prediction_vs_gt(pred, gt, decoder_config=cfg)
        m = r["metrics"]
        rows.append({
            "Config": cfg.name,
            "Mode label": cfg.label,
            "Frame F1": m.get("frame_f1", 0.0),
            "Note F1": m.get("note_f1", 0.0),
            "N+Off F1": m.get("note_with_offset_f1", 0.0),
            "N+Off+Vel F1": m.get("note_with_offset_vel_f1", 0.0),
            "Pred notes": int(m.get("n_pred_notes", 0)),
            "GT notes": int(m.get("n_gt_notes", 0)),
            "GT source": r["decoder"].get("gt_source", ""),
        })
    return pd.DataFrame(rows)
