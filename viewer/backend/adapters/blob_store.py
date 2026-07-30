"""Contained filesystem access for immutable catalog representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..services.catalog import Catalog, CatalogNotFound


class BlobNotFound(LookupError):
    pass


class UnsafeBlobPath(ValueError):
    pass


@dataclass(frozen=True)
class FileBlob:
    representation_id: str
    path: Path
    media_type: str
    byte_size: int
    sha256: str


class FileBlobStore:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def resolve(self, representation_id: str) -> FileBlob:
        try:
            record = self.catalog.get_representation(representation_id)
        except CatalogNotFound as exc:
            raise BlobNotFound(str(exc)) from exc
        root = Path(record["package_root"]).resolve()
        path = (root / record["relative_path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise UnsafeBlobPath(
                "catalog representation resolves outside its package"
            ) from exc
        if not path.is_file():
            raise BlobNotFound(
                f"representation {representation_id!r} is not a regular file"
            )
        actual_size = path.stat().st_size
        if actual_size != record["byte_size"]:
            raise UnsafeBlobPath(
                f"representation size changed after registration "
                f"({record['byte_size']} -> {actual_size})"
            )
        return FileBlob(
            representation_id=representation_id,
            path=path,
            media_type=record["media_type"],
            byte_size=record["byte_size"],
            sha256=record["sha256"],
        )
