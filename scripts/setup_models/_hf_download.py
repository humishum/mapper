"""Small Hugging Face downloader used by the model setup shell scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--filename")
    parser.add_argument("--revision")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "repo_id": args.repo_id,
        "local_dir": args.local_dir,
        "revision": args.revision,
        "force_download": args.force,
    }
    if args.filename:
        path = hf_hub_download(filename=args.filename, **common)
    else:
        path = snapshot_download(**common)
    print(path)


if __name__ == "__main__":
    main()
