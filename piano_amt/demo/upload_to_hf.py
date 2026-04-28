from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def upload_demo_package(repo_id: str, checkpoint: Path, demo_assets: Path, manifest: Path, private: bool = False) -> None:
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo="checkpoints/best.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    upload_folder(
        folder_path=str(demo_assets),
        path_in_repo="demo_assets",
        repo_id=repo_id,
        repo_type="model",
    )
    upload_file(
        path_or_fileobj=str(manifest),
        path_in_repo="demo/sample_manifest.json",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded checkpoint + demo assets to Hugging Face repo: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload checkpoint and demo assets to Hugging Face.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--demo_assets", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    upload_demo_package(args.repo_id, args.checkpoint, args.demo_assets, args.manifest, private=args.private)


if __name__ == "__main__":
    main()
