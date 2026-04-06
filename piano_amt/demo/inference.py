from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from demo.demo_config import DEFAULT_FRAME_THRESHOLD, DEFAULT_ONSET_THRESHOLD, MIDI_DIR, ensure_demo_dirs
from models.onsets_frames.decode import rolls_to_midi_file, rolls_to_note_events
from src.constants import FRAMES_PER_SECOND
from src.midi import rolls_to_midi


PredictionDict = Dict[str, torch.Tensor]


def run_model_on_mel(model, mel: torch.Tensor, device: str | torch.device | None = None) -> PredictionDict:
    """Run the model once and keep raw outputs for later slider-based decoding."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        out = model(mel.unsqueeze(0).to(device))
    return {k: v[0].detach().cpu() for k, v in out.items()}


def decode_prediction_to_pretty_midi(
    pred: PredictionDict,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
):
    return rolls_to_midi(
        onset_roll=pred["onset"],
        frame_roll=pred["frame"],
        velocity_roll=pred["velocity"],
        fps=FRAMES_PER_SECOND,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )


def prediction_to_note_events(
    pred: PredictionDict,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
):
    return rolls_to_note_events(
        onset_roll=pred["onset"],
        frame_roll=pred["frame"],
        velocity_roll=pred["velocity"],
        fps=FRAMES_PER_SECOND,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )


def save_prediction_midi(
    pred: PredictionDict,
    output_path: str | Path | None = None,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
) -> Path:
    ensure_demo_dirs()
    output_path = Path(output_path or (MIDI_DIR / "prediction.mid"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rolls_to_midi_file(
        onset_roll=pred["onset"],
        frame_roll=pred["frame"],
        velocity_roll=pred["velocity"],
        output_path=output_path,
        fps=FRAMES_PER_SECOND,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    return output_path


def count_predicted_notes(
    pred: PredictionDict,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
) -> int:
    return len(prediction_to_note_events(pred, onset_threshold, frame_threshold))
