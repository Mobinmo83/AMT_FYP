from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from demo.inference import prediction_to_note_events
from evaluate.error_analysis import compute_error_analysis
from evaluate.metrics import compute_metrics, get_eval_protocol


GroundTruthDict = Dict[str, torch.Tensor]
PredictionDict = Dict[str, torch.Tensor]


def _greedy_match_summary(
    pred: PredictionDict,
    gt: GroundTruthDict,
    onset_threshold: float,
    frame_threshold: float,
    tol_s: float = 0.05,
) -> Dict[str, float]:
    pred_events = prediction_to_note_events(pred, onset_threshold, frame_threshold)
    gt_events = prediction_to_note_events(gt, 0.5, 0.5)

    used_pred = set()
    matched_pairs = []

    for gt_e in gt_events:
        best_i = None
        best_key = (float("inf"), float("inf"))
        for i, pred_e in enumerate(pred_events):
            if i in used_pred:
                continue
            if pred_e.pitch != gt_e.pitch:
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

    matched = len(matched_pairs)
    missed = len(gt_events) - matched
    extra = len(pred_events) - matched

    onset_errs_ms = [abs(pred_e.onset_sec - gt_e.onset_sec) * 1000.0 for gt_e, pred_e in matched_pairs]
    offset_errs_ms = [abs(pred_e.offset_sec - gt_e.offset_sec) * 1000.0 for gt_e, pred_e in matched_pairs]

    return {
        "matched_notes": matched,
        "missed_notes": missed,
        "extra_notes": extra,
        "onset_timing_mae_ms": float(np.mean(onset_errs_ms)) if onset_errs_ms else 0.0,
        "offset_timing_mae_ms": float(np.mean(offset_errs_ms)) if offset_errs_ms else 0.0,
        "predicted_notes": len(pred_events),
        "ground_truth_notes": len(gt_events),
    }


def evaluate_prediction_vs_gt(
    pred: PredictionDict,
    gt: GroundTruthDict,
    onset_threshold: float,
    frame_threshold: float,
) -> Dict[str, Dict[str, float]]:
    metrics = compute_metrics(
        pred_onset=pred["onset"],
        pred_frame=pred["frame"],
        pred_offset=pred["offset"],
        pred_velocity=pred["velocity"],
        gt_onset=gt["onset"],
        gt_frame=gt["frame"],
        gt_offset=gt["offset"],
        gt_velocity=gt["velocity"],
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    error_analysis = compute_error_analysis(
        pred_onset=pred["onset"],
        pred_frame=pred["frame"],
        pred_offset=pred["offset"],
        gt_onset=gt["onset"],
        gt_frame=gt["frame"],
        gt_offset=gt["offset"],
        fps=31.25,
        onset_threshold=onset_threshold,
    )

    failure = _greedy_match_summary(pred, gt, onset_threshold, frame_threshold)
    return {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "failure": failure,
        "protocol": get_eval_protocol(),
    }


def main_scores_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    m = results["metrics"]
    rows = [
        {"Metric": "Frame F1", "Value": m.get("frame_f1", 0.0)},
        {"Metric": "Note F1 (onset+pitch)", "Value": m.get("note_f1", 0.0)},
        {"Metric": "Note w/ offset F1", "Value": m.get("note_with_offset_f1", 0.0)},
        {"Metric": "Note w/ offset + velocity F1", "Value": m.get("note_with_offset_vel_f1", 0.0)},
    ]
    return pd.DataFrame(rows)


def failure_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    ea = results["error_analysis"]
    f = results["failure"]
    rows = [
        {"Item": "Matched notes", "Value": f["matched_notes"]},
        {"Item": "Missed notes", "Value": f["missed_notes"]},
        {"Item": "Extra notes", "Value": f["extra_notes"]},
        {"Item": "Onset timing MAE (ms)", "Value": f["onset_timing_mae_ms"]},
        {"Item": "Offset timing MAE (ms)", "Value": f["offset_timing_mae_ms"]},
        {"Item": "Duplicate note rate", "Value": ea.get("duplicate_note_rate", 0.0)},
        {"Item": "Chord completeness", "Value": ea.get("chord_completeness", 0.0)},
        {"Item": "Offset MAE (error analysis, ms)", "Value": ea.get("offset_mae_ms", 0.0)},
    ]
    return pd.DataFrame(rows)
