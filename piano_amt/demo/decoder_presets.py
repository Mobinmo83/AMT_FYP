from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class DecoderPreset:
    """Named demo preset mirroring the Chapter 5 advanced-decoder ablations.

    M2 = frame smoothing
    M3 = minimum note duration
    M4 = velocity-aware duplicate removal

    The keyword dictionary intentionally contains several common aliases.  The
    adapter in ``demo.inference`` filters these against the actual
    ``decode_advanced.py`` function signature at runtime, so the public demo can
    stay compatible if your internal argument names are slightly different.
    """

    name: str
    label: str
    description: str
    advanced_kwargs: Mapping[str, object]


DEFAULT_ONSET_THRESHOLD = 0.40
DEFAULT_FRAME_THRESHOLD = 0.40

# Tuned values used for the public demo.  Keep these aligned with the values you
# report for the combined validation/test configurations in Chapter 5.
ADVANCED_DECODER_PRESETS: Dict[str, DecoderPreset] = {
    "quality_m2_m3_m4": DecoderPreset(
        name="quality_m2_m3_m4",
        label="Quality mode — M2 + M3 + M4",
        description=(
            "Full selected demo decoder: frame smoothing, minimum duration, "
            "and velocity-aware duplicate removal."
        ),
        advanced_kwargs={
            # Generic enable flags
            "use_frame_smoothing": True,
            "use_min_duration": True,
            "use_min_note_duration": True,
            "use_duplicate_removal": True,
            "use_velocity_duplicate_removal": True,
            "enable_frame_smoothing": True,
            "enable_min_duration": True,
            "enable_min_note_duration": True,
            "enable_duplicate_removal": True,
            "enable_velocity_duplicate_removal": True,
            # Explicitly keep other experimental methods off for the M2+M3+M4 table.
            "use_onset_conditioned_offsets": False,
            "use_adaptive_thresholds": False,
            "use_chord_grouping": False,
            "use_sustain_extension": False,
            "enable_onset_conditioned_offsets": False,
            "enable_adaptive_thresholds": False,
            "enable_chord_grouping": False,
            "enable_sustain_extension": False,
            # Common tuned numeric parameters / aliases.
            "frame_smoothing_method": "median",
            "smoothing_method": "median",
            "median_kernel_size": 3,
            "smooth_kernel_size": 3,
            "min_note_duration_s": 0.05,
            "min_duration_s": 0.05,
            "min_note_ms": 50.0,
            "duplicate_tolerance_s": 0.03,
            "duplicate_time_tolerance_s": 0.03,
            "duplicate_tolerance_ms": 30.0,
        },
    ),
    "efficient_m3_m4": DecoderPreset(
        name="efficient_m3_m4",
        label="Efficient mode — M3 + M4",
        description=(
            "Fast decoder for live demonstrations: minimum duration and "
            "velocity-aware duplicate removal, without frame smoothing."
        ),
        advanced_kwargs={
            "use_frame_smoothing": False,
            "enable_frame_smoothing": False,
            "use_min_duration": True,
            "use_min_note_duration": True,
            "use_duplicate_removal": True,
            "use_velocity_duplicate_removal": True,
            "enable_min_duration": True,
            "enable_min_note_duration": True,
            "enable_duplicate_removal": True,
            "enable_velocity_duplicate_removal": True,
            "use_onset_conditioned_offsets": False,
            "use_adaptive_thresholds": False,
            "use_chord_grouping": False,
            "use_sustain_extension": False,
            "enable_onset_conditioned_offsets": False,
            "enable_adaptive_thresholds": False,
            "enable_chord_grouping": False,
            "enable_sustain_extension": False,
            "min_note_duration_s": 0.05,
            "min_duration_s": 0.05,
            "min_note_ms": 50.0,
            "duplicate_tolerance_s": 0.03,
            "duplicate_time_tolerance_s": 0.03,
            "duplicate_tolerance_ms": 30.0,
        },
    ),
}


def list_decoder_modes() -> list[str]:
    return list(ADVANCED_DECODER_PRESETS.keys())


def get_decoder_preset(mode: str) -> DecoderPreset:
    if mode not in ADVANCED_DECODER_PRESETS:
        valid = ", ".join(ADVANCED_DECODER_PRESETS)
        raise KeyError(f"Unknown decoder mode {mode!r}. Valid modes: {valid}")
    return ADVANCED_DECODER_PRESETS[mode]
