#!/usr/bin/env python3
"""Install the pinned copc_converter binary after checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "tools" / "copc_converter.lock.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        help="binary path (default: .tools/copc_converter/VERSION/copc_converter)",
    )
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text())
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64"}:
        raise SystemExit(
            f"no pinned binary for {platform.system()} {platform.machine()}"
        )
    release = lock["platforms"]["linux-x86_64"]
    destination = args.destination or (
        REPO_ROOT / ".tools" / "copc_converter" / lock["version"] / "copc_converter"
    )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mapper-copc-install-") as temp_text:
        archive = Path(temp_text) / "release.tar.gz"
        with (
            urllib.request.urlopen(release["url"]) as response,
            archive.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != release["sha256"]:
            raise SystemExit(
                f"checksum mismatch: expected {release['sha256']}, got {digest}"
            )
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember(release["archive_member"])
            if not member.isfile() or Path(member.name).name != member.name:
                raise SystemExit(f"unsafe archive member: {member.name!r}")
            source = tar.extractfile(member)
            if source is None:
                raise SystemExit(f"missing archive member: {member.name!r}")
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            temporary.chmod(0o755)
            os.replace(temporary, destination)

    print(f"installed copc_converter {lock['version']} at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
