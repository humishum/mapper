"""Validation gate for reconstruction packages before catalog registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from ..domain.package import (
    Manifest,
    Metrics,
    SourcesDocument,
    TABULAR_COLUMN_DTYPES,
)

BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None


class PackageValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__(
            "; ".join(
                f"{issue.path + ': ' if issue.path else ''}{issue.message}"
                for issue in self.issues
            )
        )


@dataclass(frozen=True)
class ValidatedPackage:
    root: Path
    manifest_path: Path
    manifest: Manifest
    manifest_sha256: str
    sources: SourcesDocument | None
    metrics: Metrics | None

    def artifact_path(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve(strict=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash a directory deterministically using relative paths and file contents."""
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"directory representation contains a symlink: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


class PackageValidator:
    """Validate structure, contracts, containment, sizes, and checksums."""

    def __init__(self, *, verify_checksums: bool = True):
        self.verify_checksums = verify_checksums

    def validate(self, package_root: str | Path) -> ValidatedPackage:
        root = Path(package_root).resolve()
        manifest_path = root / "manifest.json"
        issues: list[ValidationIssue] = []

        if not root.is_dir():
            raise PackageValidationError(
                [
                    ValidationIssue(
                        "package_not_found", "package root is not a directory"
                    )
                ]
            )
        if not manifest_path.is_file():
            raise PackageValidationError(
                [
                    ValidationIssue(
                        "manifest_missing",
                        "required manifest.json is missing",
                        "manifest.json",
                    )
                ]
            )

        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PackageValidationError(
                [ValidationIssue("manifest_invalid_json", str(exc), "manifest.json")]
            ) from exc
        try:
            manifest = Manifest.model_validate(manifest_data)
        except ValidationError as exc:
            raise PackageValidationError(
                [
                    ValidationIssue(
                        "manifest_contract",
                        error["msg"],
                        ".".join(str(part) for part in error["loc"]),
                    )
                    for error in exc.errors()
                ]
            ) from exc

        sources: SourcesDocument | None = None
        metrics: Metrics | None = None
        for artifact in manifest.artifacts:
            relative_path = artifact.path
            candidate = (root / relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                issues.append(
                    ValidationIssue(
                        "path_escape",
                        "artifact resolves outside the package root",
                        relative_path,
                    )
                )
                continue
            if not candidate.exists():
                issues.append(
                    ValidationIssue(
                        "artifact_missing",
                        "declared artifact is missing",
                        relative_path,
                    )
                )
                continue

            actual_size = (
                candidate.stat().st_size
                if candidate.is_file()
                else sum(
                    child.stat().st_size
                    for child in candidate.rglob("*")
                    if child.is_file() and not child.is_symlink()
                )
            )
            if actual_size != artifact.byte_size:
                issues.append(
                    ValidationIssue(
                        "byte_size_mismatch",
                        f"declared {artifact.byte_size}, found {actual_size}",
                        relative_path,
                    )
                )
            if self.verify_checksums:
                try:
                    actual_sha256 = (
                        sha256_file(candidate)
                        if candidate.is_file()
                        else sha256_directory(candidate)
                    )
                except ValueError as exc:
                    issues.append(
                        ValidationIssue("unsafe_directory", str(exc), relative_path)
                    )
                else:
                    if actual_sha256 != artifact.sha256:
                        issues.append(
                            ValidationIssue(
                                "checksum_mismatch",
                                f"declared {artifact.sha256}, found {actual_sha256}",
                                relative_path,
                            )
                        )

            if artifact.kind == "sources" and candidate.is_file():
                sources = self._validate_json_sidecar(
                    candidate, SourcesDocument, issues, relative_path
                )
                if sources is not None:
                    for source in sources.sources:
                        if source.run_id != manifest.run_id:
                            issues.append(
                                ValidationIssue(
                                    "source_run_mismatch",
                                    f"source {source.source_index} refers to another run",
                                    relative_path,
                                )
                            )
                        if source.capture_id != manifest.capture_id:
                            issues.append(
                                ValidationIssue(
                                    "source_capture_mismatch",
                                    f"source {source.source_index} refers to another capture",
                                    relative_path,
                                )
                            )
            elif artifact.kind == "metrics" and candidate.is_file():
                metrics = self._validate_json_sidecar(
                    candidate, Metrics, issues, relative_path
                )
            if artifact.kind in TABULAR_COLUMN_DTYPES:
                self._validate_parquet_schema(
                    candidate, artifact.columns, issues, relative_path
                )

        kinds = [artifact.kind for artifact in manifest.artifacts]
        if kinds.count("sources") > 1 or kinds.count("metrics") > 1:
            issues.append(
                ValidationIssue(
                    "duplicate_sidecar",
                    "a package may contain at most one sources and one metrics document",
                )
            )
        if (
            any(
                "PointSourceId" in artifact.required_dimensions
                or "SourceIndex" in artifact.required_dimensions
                for artifact in manifest.artifacts
                if artifact.kind == "points"
            )
            and sources is None
        ):
            issues.append(
                ValidationIssue(
                    "sources_required",
                    "points with a provenance dimension require a valid sources document",
                )
            )

        if issues:
            raise PackageValidationError(issues)
        return ValidatedPackage(
            root=root,
            manifest_path=manifest_path,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            sources=sources,
            metrics=metrics,
        )

    @staticmethod
    def _validate_json_sidecar(path, model, issues, relative_path):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    "sidecar_unreadable", str(exc), relative_path
                )
            )
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    "sidecar_invalid_json",
                    exc.msg,
                    f"{relative_path}:line {exc.lineno}:column {exc.colno}",
                )
            )
            return None
        try:
            return model.model_validate(value)
        except ValidationError as exc:
            issues.extend(
                ValidationIssue(
                    "sidecar_contract",
                    error["msg"],
                    ".".join(
                        [relative_path, *(str(part) for part in error["loc"])]
                    ),
                )
                for error in exc.errors()
            )
            return None

    @staticmethod
    def _validate_parquet_schema(path, columns, issues, relative_path) -> None:
        if not path.is_file():
            issues.append(
                ValidationIssue(
                    "parquet_not_file",
                    "canonical tabular artifacts must be a single Parquet file",
                    relative_path,
                )
            )
            return
        try:
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    "parquet_unreadable",
                    f"cannot read Parquet footer/schema: {exc}",
                    relative_path,
                )
            )
            return

        for column in columns:
            indexes = schema.get_all_field_indices(column.name)
            if not indexes:
                issues.append(
                    ValidationIssue(
                        "parquet_column_missing",
                        f"declared column {column.name!r} is absent from the file",
                        relative_path,
                    )
                )
                continue
            if len(indexes) > 1:
                issues.append(
                    ValidationIssue(
                        "parquet_column_duplicate",
                        f"column {column.name!r} occurs more than once",
                        relative_path,
                    )
                )
                continue
            physical_dtype = _canonical_arrow_dtype(schema.field(indexes[0]).type)
            if physical_dtype != column.dtype:
                issues.append(
                    ValidationIssue(
                        "parquet_dtype_mismatch",
                        f"column {column.name!r} declares {column.dtype!r} "
                        f"but the file contains {physical_dtype!r}",
                        relative_path,
                    )
                )


def _canonical_arrow_dtype(dtype: pa.DataType) -> str:
    """Return stable manifest dtype names instead of Arrow display aliases."""
    aliases = (
        (pa.types.is_float64, "float64"),
        (pa.types.is_float32, "float32"),
        (pa.types.is_int64, "int64"),
        (pa.types.is_int32, "int32"),
        (pa.types.is_int16, "int16"),
        (pa.types.is_int8, "int8"),
        (pa.types.is_uint64, "uint64"),
        (pa.types.is_uint32, "uint32"),
        (pa.types.is_uint16, "uint16"),
        (pa.types.is_uint8, "uint8"),
        (pa.types.is_boolean, "bool"),
        (pa.types.is_string, "utf8"),
        (pa.types.is_large_string, "large_utf8"),
        (pa.types.is_binary, "binary"),
        (pa.types.is_large_binary, "large_binary"),
    )
    for predicate, name in aliases:
        if predicate(dtype):
            return name
    return str(dtype)
