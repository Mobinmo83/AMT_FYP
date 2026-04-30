"""
demo decoder configuration — preset definitions for baseline and advanced decoding.

Purpose:
  This file defines the editable decoder presets used by the public demo.
  Each preset controls how prediction rolls are converted into note events
  and MIDI output after model inference.

Design:
  - AdvancedDecoderConfig stores the decode thresholds, metadata, and
    post-processing toggles in one frozen dataclass.
  - Field names match the advanced decoder function arguments, so presets can
    be passed directly into the final decoding path without alias conversion.
  - ADVANCED_DECODER_PRESETS provides named demo modes for baseline decoding,
    efficient post-processing, quality post-processing, and optional
    single-method explanations.
  - make_decoder_config() loads a preset and applies optional overrides for
    notebook controls or interactive demo settings.
  - config_table_dict() converts a config into a plain dictionary for display.

Demo modes:
  - baseline: tuned baseline decoder with no post-processing.
  - efficient_m3_m4: minimum note duration + velocity-aware duplicate removal.
  - quality_m2_m3_m4: frame smoothing + minimum duration + duplicate removal.
  - m2_only / m3_only / m4_only: single-method modes for live explanation.

Outputs:
  - Decoder configuration objects used by inference, MIDI generation, plots,
    and demo result summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class AdvancedDecoderConfig:
    """Editable decoder configuration used by the public demo.

    Field names match ``models.onsets_frames.decode_advanced.advanced_rolls_to_note_events``.
    This prevents alias filtering and keeps the demo aligned with the final
    evaluation notebook.

    M2 = frame-level smoothing
    M3 = minimum note duration
    M4 = velocity-aware duplicate removal
    """

    name: str
    label: str
    description: str
    decoder_type: str = "advanced"  # "advanced" or "baseline"

    # Final/test decode thresholds for prediction rolls.
    onset_threshold: float = 0.40
    frame_threshold: float = 0.40
    offset_threshold: float = 0.50

    # M1: onset-conditioned offset estimation.
    use_onset_conditioned_offset: bool = False

    # M2: frame-level smoothing.
    use_frame_smoothing: bool = False
    frame_smoothing_kernel: int = 7
    frame_smoothing_method: str = "median"  # "median", "gaussian", or "closing"

    # M3: minimum note duration.
    min_note_duration_ms: float = 16.0

    # M4: velocity-aware duplicate removal.
    use_duplicate_removal: bool = False
    duplicate_tolerance_sec: float = 0.05

    # M5: chord-aware onset grouping.
    use_chord_grouping: bool = False
    chord_tolerance_sec: float = 0.03
    chord_snap_to: str = "median"

    # M6: adaptive thresholds.
    use_adaptive_thresholds: bool = False
    adaptive_onset_k: float = 0.5
    adaptive_frame_k: float = 0.5

    # M7: sustain-pedal-aware extension.
    use_pedal_extension: bool = False
    pedal_energy_threshold: float = 10.0
    pedal_max_extension_sec: float = 2.0

    def decoder_kwargs(self) -> Dict[str, Any]:
        """Return only kwargs consumed by the advanced decoder.

        Thresholds and metadata fields are removed because thresholds are passed
        explicitly, matching the final evaluation notebook call pattern.
        """
        d = asdict(self)
        for key in [
            "name", "label", "description", "decoder_type",
            "onset_threshold", "frame_threshold", "offset_threshold",
        ]:
            d.pop(key, None)
        return d

    def with_overrides(self, **overrides: Any) -> "AdvancedDecoderConfig":
        valid = set(asdict(self).keys())
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise KeyError(f"Unknown decoder config field(s): {unknown}")
        return replace(self, **overrides)


ADVANCED_DECODER_PRESETS: Dict[str, AdvancedDecoderConfig] = {
    "baseline": AdvancedDecoderConfig(
        name="adv_baseline",
        label="Tuned baseline — no post-processing",
        description=(
            "No post-processing. Original onset-gated baseline decoder at the "
            "tuned prediction thresholds 0.4/0.4."
        ),
        decoder_type="baseline",
        onset_threshold=0.40,
        frame_threshold=0.40,
        offset_threshold=0.50,
        min_note_duration_ms=16.0,
    ),
    "efficient_m3_m4": AdvancedDecoderConfig(
        name="adv_m3_m4",
        label="Efficient mode — M3 + M4",
        description="Minimum note duration + velocity-aware duplicate removal.",
        min_note_duration_ms=55.0,
        use_duplicate_removal=True,
        duplicate_tolerance_sec=0.06,
    ),
    "quality_m2_m3_m4": AdvancedDecoderConfig(
        name="adv_m2_m3_m4",
        label="Quality mode — M2 + M3 + M4",
        description=(
            "Frame-level closing smoothing + minimum note duration + "
            "velocity-aware duplicate removal."
        ),
        use_frame_smoothing=True,
        frame_smoothing_kernel=3,
        frame_smoothing_method="closing",
        min_note_duration_ms=55.0,
        use_duplicate_removal=True,
        duplicate_tolerance_sec=0.06,
    ),
    # Optional single-method modes for live explanation.
    "m2_only": AdvancedDecoderConfig(
        name="demo_m2_only",
        label="M2 only — frame smoothing",
        description="Only frame-level closing smoothing is enabled.",
        use_frame_smoothing=True,
        frame_smoothing_kernel=3,
        frame_smoothing_method="closing",
        min_note_duration_ms=16.0,
    ),
    "m3_only": AdvancedDecoderConfig(
        name="demo_m3_only",
        label="M3 only — minimum duration",
        description="Only minimum note duration is enabled.",
        min_note_duration_ms=55.0,
    ),
    "m4_only": AdvancedDecoderConfig(
        name="demo_m4_only",
        label="M4 only — duplicate removal",
        description="Only velocity-aware duplicate removal is enabled.",
        use_duplicate_removal=True,
        duplicate_tolerance_sec=0.06,
    ),
}

DEFAULT_MODE = "quality_m2_m3_m4"
DEFAULT_ONSET_THRESHOLD = 0.40
DEFAULT_FRAME_THRESHOLD = 0.40
DEFAULT_OFFSET_THRESHOLD = 0.50


def list_decoder_modes(include_single_methods: bool = True) -> list[str]:
    if include_single_methods:
        return list(ADVANCED_DECODER_PRESETS.keys())
    return ["baseline", "efficient_m3_m4", "quality_m2_m3_m4"]


def get_decoder_preset(mode: str = DEFAULT_MODE) -> AdvancedDecoderConfig:
    if mode not in ADVANCED_DECODER_PRESETS:
        valid = ", ".join(ADVANCED_DECODER_PRESETS)
        raise KeyError(f"Unknown decoder mode {mode!r}. Valid modes: {valid}")
    return ADVANCED_DECODER_PRESETS[mode]


def make_decoder_config(
    mode: str = DEFAULT_MODE,
    overrides: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> AdvancedDecoderConfig:
    merged: Dict[str, Any] = {}
    if overrides:
        merged.update(dict(overrides))
    merged.update(kwargs)
    return get_decoder_preset(mode).with_overrides(**merged) if merged else get_decoder_preset(mode)


def config_table_dict(cfg: AdvancedDecoderConfig) -> Dict[str, Any]:
    return asdict(cfg)
