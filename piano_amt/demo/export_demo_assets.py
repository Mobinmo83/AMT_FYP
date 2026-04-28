from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pretty_midi


def _first_existing(root: Path, relative_path: str | None) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path:
        return None
    p = root / relative_path
    return p if p.exists() else None


def _duration_from_audio(audio_path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def _midi_note_summary(midi_path: Path | None) -> dict:
    if midi_path is None or not midi_path.exists():
        return {}
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes if 21 <= n.pitch <= 108]
        cc64 = [cc for inst in pm.instruments for cc in inst.control_changes if cc.number == 64]
        return {
            "midi_note_count": len(notes),
            "midi_duration_s": float(pm.get_end_time()),
            "has_sustain_cc64": bool(cc64),
            "sustain_cc64_count": len(cc64),
        }
    except Exception as exc:
        return {"midi_summary_error": str(exc)}


def export_demo_assets(
    maestro_root: Path,
    cache_dir: Path,
    output_dir: Path,
    names: list[str],
    manifest_path: Path | None = None,
) -> None:
    """Export public demo audio, cached labels, original MIDI, and metadata.

    The previous version exported only wav + label rolls.  This version also
    copies the original MAESTRO MIDI so the demo can play/download the true
    reference performance and evaluate note events against it when requested.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(maestro_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No MAESTRO CSV found in {maestro_root}")
    df = pd.read_csv(csv_files[0])

    samples = []
    for i, stem in enumerate(names, start=1):
        matches = df[df["audio_filename"].astype(str).str.contains(stem, regex=False)].head(1)
        if matches.empty:
            matches = df[df.get("midi_filename", "").astype(str).str.contains(stem, regex=False)].head(1)
        if matches.empty:
            raise KeyError(f"Could not find audio/MIDI stem in MAESTRO CSV: {stem}")
        row = matches.iloc[0]

        audio_path = _first_existing(maestro_root, row.get("audio_filename"))
        midi_path = _first_existing(maestro_root, row.get("midi_filename"))
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for stem {stem}: {row.get('audio_filename')}")
        if midi_path is None:
            raise FileNotFoundError(f"Original MIDI file not found for stem {stem}: {row.get('midi_filename')}")

        cache_stems = [Path(str(row.get("audio_filename"))).stem, Path(str(row.get("midi_filename"))).stem]
        cache_path = next((cache_dir / f"{s}.npz" for s in cache_stems if (cache_dir / f"{s}.npz").exists()), None)
        if cache_path is None:
            raise FileNotFoundError(f"Cache file not found for any of: {cache_stems} in {cache_dir}")

        target_audio = output_dir / f"sample_{i:02d}{audio_path.suffix.lower()}"
        target_labels = output_dir / f"sample_{i:02d}_labels.npz"
        target_midi = output_dir / f"sample_{i:02d}_original.mid"
        shutil.copy2(audio_path, target_audio)
        shutil.copy2(midi_path, target_midi)

        data = np.load(str(cache_path))
        np.savez_compressed(
            target_labels,
            onset=data["onset"],
            frame=data["frame"],
            offset=data["offset"],
            velocity=data["velocity"],
        )

        metadata = {
            "source_audio_filename": str(row.get("audio_filename", "")),
            "source_midi_filename": str(row.get("midi_filename", "")),
            "split": str(row.get("split", "")),
            "canonical_composer": str(row.get("canonical_composer", "")),
            "canonical_title": str(row.get("canonical_title", "")),
            "year": str(row.get("year", "")),
            "duration_s_audio": _duration_from_audio(audio_path),
            "label_frames": int(data["frame"].shape[0]),
            "label_fps": 31.25,
            **_midi_note_summary(midi_path),
        }

        samples.append(
            {
                "name": f"MAESTRO Test {i:02d} — {Path(row['audio_filename']).stem}",
                "audio": f"demo_assets/{target_audio.name}",
                "labels": f"demo_assets/{target_labels.name}",
                "midi": f"demo_assets/{target_midi.name}",
                "metadata": metadata,
            }
        )

    manifest = {"samples": samples}
    manifest_path = manifest_path or (output_dir.parent / "demo" / "sample_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest → {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a tiny public MAESTRO demo subset.")
    parser.add_argument("--maestro_root", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--names", nargs="+", required=True, help="One or more audio or MIDI stems to export")
    args = parser.parse_args()
    export_demo_assets(args.maestro_root, args.cache_dir, args.output_dir, args.names, args.manifest_path)


if __name__ == "__main__":
    main()
