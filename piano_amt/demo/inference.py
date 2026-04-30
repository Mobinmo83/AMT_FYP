"""
demo inference — full-length model prediction, note-event decoding, and MIDI output.

Purpose:
  This file handles the inference stage of the public demo. It runs the loaded
  model on a log-mel spectrogram, decodes prediction rolls into note events
  using the selected demo decoder mode, and saves predicted or ground-truth
  MIDI files for listening, visualisation, and comparison.

Design:
  - run_model_on_mel() performs full-length single-pass inference and returns
    onset, frame, offset, and velocity prediction rolls on CPU.
  - DemoNoteEvent provides a compact event format used across demo evaluation,
    plotting, MIDI writing, and comparison utilities.
  - prediction_to_note_events() decodes model predictions through either the
    baseline decoder or the advanced decoder preset.
  - gt_rolls_to_note_events() decodes cached ground-truth rolls at fixed
    thresholds so the reference event set remains stable.
  - note_events_to_pretty_midi() converts decoded events into a PrettyMIDI
    object.
  - save_prediction_midi() and save_gt_eval_midi() write MIDI files into the
    demo output directory.

Outputs:
  - Prediction dictionaries containing model output rolls.
  - Decoded note-event lists for evaluation and plotting.
  - PrettyMIDI objects for predicted and reference transcriptions.
  - Saved .mid files for demo playback, download, and qualitative inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pretty_midi
import torch

from demo.decoder_presets import AdvancedDecoderConfig, DEFAULT_MODE, make_decoder_config
from demo.demo_config import MIDI_DIR, ensure_demo_dirs
from models.onsets_frames.decode import rolls_to_note_events
from models.onsets_frames.decode_advanced import advanced_rolls_to_note_events
from src.constants import FRAMES_PER_SECOND

PredictionDict = Dict[str, torch.Tensor]
GroundTruthDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class DemoNoteEvent:
    pitch: int
    onset_sec: float
    offset_sec: float
    velocity: int = 64


def run_model_on_mel(
    model: torch.nn.Module,
    mel: torch.Tensor,
    device: str | torch.device | None = None,
) -> PredictionDict:
    """Run full-length single-pass inference, matching evaluate_advanced.py."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        out = model(mel.unsqueeze(0).to(device))
    return {k: v[0].detach().cpu() for k, v in out.items()}


def _to_demo_events(events: Iterable) -> list[DemoNoteEvent]:
    demo_events: list[DemoNoteEvent] = []
    for e in events:
        pitch = int(getattr(e, "pitch"))
        onset = float(getattr(e, "onset_sec"))
        offset = float(getattr(e, "offset_sec"))
        velocity = int(np.clip(round(float(getattr(e, "velocity", 64))), 1, 127))
        if offset <= onset:
            offset = onset + 1.0 / FRAMES_PER_SECOND
        demo_events.append(DemoNoteEvent(pitch=pitch, onset_sec=onset, offset_sec=offset, velocity=velocity))
    demo_events.sort(key=lambda e: (e.onset_sec, e.pitch, e.offset_sec))
    return demo_events


def prediction_to_note_events(
    pred: PredictionDict,
    decoder_config: AdvancedDecoderConfig | None = None,
    decoder_mode: str = DEFAULT_MODE,
    **decoder_overrides,
) -> list[DemoNoteEvent]:
    """Decode predictions through the selected public-demo decoder.

    ``baseline`` uses the original baseline decoder. The final efficient/quality
    modes use ``decode_advanced.py`` directly. There is deliberately no silent
    fallback from advanced mode to baseline.
    """
    cfg = decoder_config or make_decoder_config(decoder_mode, **decoder_overrides)

    if cfg.decoder_type == "baseline":
        events = rolls_to_note_events(
            onset_roll=pred["onset"],
            frame_roll=pred["frame"],
            velocity_roll=pred["velocity"],
            fps=FRAMES_PER_SECOND,
            onset_threshold=cfg.onset_threshold,
            frame_threshold=cfg.frame_threshold,
        )
        return _to_demo_events(events)

    events = advanced_rolls_to_note_events(
        onset_roll=pred["onset"],
        frame_roll=pred["frame"],
        offset_roll=pred["offset"],
        velocity_roll=pred["velocity"],
        fps=FRAMES_PER_SECOND,
        onset_threshold=cfg.onset_threshold,
        frame_threshold=cfg.frame_threshold,
        offset_threshold=cfg.offset_threshold,
        **cfg.decoder_kwargs(),
    )
    return _to_demo_events(events)


def gt_rolls_to_note_events(gt: GroundTruthDict) -> list[DemoNoteEvent]:
    """Decode cached GT rolls exactly as evaluate_advanced.py does.

    The prediction threshold can be tuned to 0.4/0.4, but the cached binary
    GT labels are always decoded at 0.5/0.5 so the reference event set is fixed.
    """
    events = rolls_to_note_events(
        onset_roll=gt["onset"],
        frame_roll=gt["frame"],
        velocity_roll=gt["velocity"],
        fps=FRAMES_PER_SECOND,
        onset_threshold=0.5,
        frame_threshold=0.5,
    )
    return _to_demo_events(events)


def note_events_to_pretty_midi(events: Iterable[DemoNoteEvent], program: int = 0, name: str = "Piano") -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program), is_drum=False, name=name)
    for e in events:
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(np.clip(e.velocity, 1, 127)),
                pitch=int(np.clip(e.pitch, 0, 127)),
                start=max(0.0, float(e.onset_sec)),
                end=max(float(e.offset_sec), float(e.onset_sec) + 1e-3),
            )
        )
    inst.notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    pm.instruments.append(inst)
    return pm


def decode_prediction_to_pretty_midi(
    pred: PredictionDict,
    decoder_config: AdvancedDecoderConfig | None = None,
    decoder_mode: str = DEFAULT_MODE,
    program: int = 0,
    **decoder_overrides,
) -> pretty_midi.PrettyMIDI:
    events = prediction_to_note_events(pred, decoder_config=decoder_config, decoder_mode=decoder_mode, **decoder_overrides)
    cfg = decoder_config or make_decoder_config(decoder_mode, **decoder_overrides)
    return note_events_to_pretty_midi(events, program=program, name=f"Predicted {cfg.name}")


def gt_rolls_to_pretty_midi(gt: GroundTruthDict, program: int = 0) -> pretty_midi.PrettyMIDI:
    return note_events_to_pretty_midi(gt_rolls_to_note_events(gt), program=program, name="GT evaluation MIDI from cached rolls")


def save_prediction_midi(
    pred: PredictionDict,
    output_path: str | Path | None = None,
    decoder_config: AdvancedDecoderConfig | None = None,
    decoder_mode: str = DEFAULT_MODE,
    program: int = 0,
    **decoder_overrides,
) -> Path:
    ensure_demo_dirs()
    cfg = decoder_config or make_decoder_config(decoder_mode, **decoder_overrides)
    output_path = Path(output_path or (MIDI_DIR / f"prediction_{cfg.name}.mid"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm = decode_prediction_to_pretty_midi(pred, decoder_config=cfg, program=program)
    pm.write(str(output_path))
    return output_path


def save_gt_eval_midi(gt: GroundTruthDict, output_path: str | Path | None = None, program: int = 0) -> Path:
    ensure_demo_dirs()
    output_path = Path(output_path or (MIDI_DIR / "ground_truth_eval_rolls.mid"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm = gt_rolls_to_pretty_midi(gt, program=program)
    pm.write(str(output_path))
    return output_path


def count_predicted_notes(
    pred: PredictionDict,
    decoder_config: AdvancedDecoderConfig | None = None,
    decoder_mode: str = DEFAULT_MODE,
    **decoder_overrides,
) -> int:
    return len(prediction_to_note_events(pred, decoder_config=decoder_config, decoder_mode=decoder_mode, **decoder_overrides))
