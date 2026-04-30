"""
upload_demo_package.py — Upload the demo checkpoint and assets package(Hugging Face).

Purpose:
  This file uploads the public demo package to a model repository so the demo
  notebook can download the checkpoint, prepared audio examples, label files,
  original MIDI files, and sample manifest from one consistent location.

Design:
  - create_repo() creates the target model repository if it does not already
    exist.
  - upload_demo_package() uploads the trained checkpoint to checkpoints/best.pt.
  - The prepared demo_assets folder is uploaded as a complete asset directory.
  - sample_manifest.json is uploaded to demo/sample_manifest.json so the demo
    loader can find prepared examples.
  - An optional README can be uploaded when provided.
  - The --private flag controls whether the target repository is created as
    private.

Usage:
    python upload_demo_package.py \\
        --repo_id username/repo-name \\
        --checkpoint /path/to/best.pt \\
        --demo_assets /path/to/demo_assets \\
        --manifest /path/to/sample_manifest.json \\
        --readme /path/to/README.md

Outputs:
  - Uploaded checkpoint file.
  - Uploaded demo asset folder.
  - Uploaded sample manifest.
  - Optional uploaded README.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import create_repo, upload_file, upload_folder


def upload_demo_package(
    repo_id: str,
    checkpoint: Path,
    demo_assets: Path,
    manifest: Path,
    private: bool = False,
    readme: Path | None = None,
) -> None:
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    upload_file(path_or_fileobj=str(checkpoint), path_in_repo="checkpoints/best.pt", repo_id=repo_id, repo_type="model")
    upload_folder(folder_path=str(demo_assets), path_in_repo="demo_assets", repo_id=repo_id, repo_type="model")
    upload_file(path_or_fileobj=str(manifest), path_in_repo="demo/sample_manifest.json", repo_id=repo_id, repo_type="model")
    if readme is not None and Path(readme).exists():
        upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    print(f"Uploaded checkpoint + demo assets + manifest to Hugging Face repo: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload checkpoint and demo assets to Hugging Face.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--demo_assets", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    upload_demo_package(args.repo_id, args.checkpoint, args.demo_assets, args.manifest, args.private, args.readme)


if __name__ == "__main__":
    main()
