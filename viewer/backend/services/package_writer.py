"""Small, atomic helpers for assembling a package without duplicating contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..domain.package import ArtifactFile, Manifest
from .package_validator import sha256_directory, sha256_file


def _contained_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("package path escapes the package root") from exc
    return target


def write_json_sidecar(
    package_root: str | Path,
    relative_path: str,
    document: BaseModel,
) -> Path:
    """Write a Pydantic sidecar atomically, returning its final path."""
    root = Path(package_root)
    target = _contained_path(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        document.model_dump_json(indent=2, by_alias=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def describe_artifact(
    package_root: str | Path,
    relative_path: str,
    *,
    representation_id: str,
    kind: str,
    format: str,
    media_type: str,
    **contract_fields: Any,
) -> ArtifactFile:
    """Create a checksummed manifest entry for an already-written file/tree."""
    root = Path(package_root).resolve()
    target = _contained_path(root, relative_path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_file():
        byte_size = target.stat().st_size
        checksum = sha256_file(target)
    elif target.is_dir():
        byte_size = sum(
            child.stat().st_size
            for child in target.rglob("*")
            if child.is_file() and not child.is_symlink()
        )
        checksum = sha256_directory(target)
    else:
        raise ValueError(f"artifact is neither a regular file nor directory: {target}")
    return ArtifactFile(
        representation_id=representation_id,
        kind=kind,
        format=format,
        path=relative_path,
        media_type=media_type,
        byte_size=byte_size,
        sha256=checksum,
        **contract_fields,
    )


def write_manifest(package_root: str | Path, manifest: Manifest) -> Path:
    """Write manifest.json last and atomically, making package visibility explicit."""
    return write_json_sidecar(package_root, "manifest.json", manifest)
