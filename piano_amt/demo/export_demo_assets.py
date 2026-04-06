from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def export_demo_assets(maestro_root: Path, cache_dir: Path, output_dir: Path, names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(maestro_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No MAESTRO CSV found in {maestro_root}")
    df = pd.read_csv(csv_files[0])

    samples = []
    for i, stem in enumerate(names, start=1):
        matches = df[df["audio_filename"].str.contains(stem, regex=False)].head(1)
        if matches.empty:
            raise KeyError(f"Could not find audio stem in MAESTRO CSV: {stem}")
        row = matches.iloc[0]
        audio_path = maestro_root / row["audio_filename"]
        cache_path = cache_dir / f"{Path(row['audio_filename']).stem}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        target_audio = output_dir / f"sample_{i:02d}.wav"
        target_labels = output_dir / f"sample_{i:02d}_labels.npz"
        shutil.copy2(audio_path, target_audio)

        data = np.load(str(cache_path))
        np.savez_compressed(
            target_labels,
            onset=data["onset"],
            frame=data["frame"],
            offset=data["offset"],
            velocity=data["velocity"],
        )

        samples.append({
            "name": f"MAESTRO Test {i:02d} — {Path(row['audio_filename']).stem}",
            "audio": f"demo_assets/{target_audio.name}",
            "labels": f"demo_assets/{target_labels.name}",
        })

    manifest = {"samples": samples}
    manifest_path = output_dir.parent / "demo" / "sample_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest → {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a tiny public MAESTRO demo subset.")
    parser.add_argument("--maestro_root", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--names", nargs="+", required=True, help="One or more audio stems to export")
    args = parser.parse_args()
    export_demo_assets(args.maestro_root, args.cache_dir, args.output_dir, args.names)


if __name__ == "__main__":
    main()
