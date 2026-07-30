"""Out-of-core COPC publication through pinned ``copc_converter``."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .copc_validation import (
    CopcStructure,
    flatten_copc_hierarchy,
    validate_copc_structure,
)

COPC_CONVERTER_VERSION = "0.11.0"
COPC_CONVERTER_ENV = "MAPPER_COPC_CONVERTER"
REPO_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / ".tools"
    / "copc_converter"
    / COPC_CONVERTER_VERSION
    / "copc_converter"
)


class CopcPublisherError(RuntimeError):
    """COPC publication failed validation or the converter failed."""


@dataclass(frozen=True)
class CopcPublisherConfig:
    executable: Path | None = None
    memory_limit: str = "4G"
    threads: int | None = None
    temp_dir: Path | None = None
    temp_compression: Literal["none", "lz4"] = "lz4"
    node_storage: Literal["files", "packed"] = "packed"
    progress: Literal["bar", "plain", "json"] = "json"
    required_dimensions: tuple[str, ...] = ("ContributorCount",)
    expected_scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    expected_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    validate_source_distribution: bool = True
    flatten_hierarchy: bool = True
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[1-9][0-9]*(?:[KMGTP])?", self.memory_limit):
            raise ValueError(
                "memory_limit must be an integer with an optional K/M/G/T/P suffix"
            )
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be positive")


@dataclass(frozen=True)
class PublishResult:
    output: Path
    point_count: int
    file_bytes: int
    sha256: str
    wall_seconds: float
    converter_version: str
    input_files: tuple[Path, ...]
    source_distribution: dict[int, int]
    structure: CopcStructure
    stdout: str
    stderr: str


@dataclass(frozen=True)
class _InputSummary:
    files: tuple[Path, ...]
    point_count: int
    dimensions: tuple[str, ...]
    point_format: int
    source_distribution: dict[int, int]


def _laspy():
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise CopcPublisherError(
            "COPC publication requires the project's laspy and lazrs dependencies"
        ) from exc
    return laspy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CopcPublisher:
    """Publish canonical LAZ shards as one validated, atomically-written COPC."""

    def __init__(self, config: CopcPublisherConfig | None = None):
        self.config = config or CopcPublisherConfig()

    def resolve_executable(self) -> Path:
        candidates = []
        if self.config.executable is not None:
            candidates.append(Path(self.config.executable))
        if os.environ.get(COPC_CONVERTER_ENV):
            candidates.append(Path(os.environ[COPC_CONVERTER_ENV]))
        candidates.append(REPO_TOOL_PATH)
        path_binary = shutil.which("copc_converter")
        if path_binary:
            candidates.append(Path(path_binary))
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        raise CopcPublisherError(
            "copc_converter was not found. Run "
            "`python scripts/install_copc_converter.py` or set "
            f"{COPC_CONVERTER_ENV}."
        )

    def _check_version(self, executable: Path) -> str:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        version_line = (result.stdout or result.stderr).strip()
        if result.returncode:
            raise CopcPublisherError(
                f"could not query {executable}: {version_line or result.returncode}"
            )
        match = re.search(r"\b([0-9]+\.[0-9]+\.[0-9]+)\b", version_line)
        version = match.group(1) if match else ""
        if version != COPC_CONVERTER_VERSION:
            raise CopcPublisherError(
                f"copc_converter {COPC_CONVERTER_VERSION} is required, "
                f"but {executable} reports {version_line!r}"
            )
        return version

    @staticmethod
    def _discover(inputs: Path | Sequence[Path]) -> tuple[Path, ...]:
        if isinstance(inputs, (str, os.PathLike)):
            input_path = Path(inputs)
            if input_path.is_dir():
                files = tuple(
                    sorted(
                        path.resolve()
                        for path in input_path.iterdir()
                        if path.is_file()
                        and path.name.lower().endswith((".las", ".laz"))
                    )
                )
            else:
                files = (input_path.resolve(),)
        else:
            files = tuple(Path(path).resolve() for path in inputs)
        if not files:
            raise CopcPublisherError("no LAS/LAZ input shards were found")
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise CopcPublisherError(f"input shards do not exist: {missing}")
        invalid = [
            str(path)
            for path in files
            if not path.name.lower().endswith((".las", ".laz"))
        ]
        if invalid:
            raise CopcPublisherError(f"inputs must be LAS/LAZ files: {invalid}")
        if len(set(files)) != len(files):
            raise CopcPublisherError("input shard list contains duplicates")
        return files

    def _summarize_inputs(self, files: tuple[Path, ...]) -> _InputSummary:
        laspy = _laspy()
        point_count = 0
        schema: tuple[str, ...] | None = None
        point_format: int | None = None
        source_counts = np.zeros(65536, dtype=np.int64)
        for path in files:
            try:
                with laspy.open(path) as reader:
                    header = reader.header
                    dimensions = tuple(header.point_format.dimension_names)
                    if str(header.version) != "1.4" or header.point_format.id not in {
                        6,
                        7,
                    }:
                        raise CopcPublisherError(
                            f"{path} must be LAS 1.4 point format 6 or 7; got "
                            f"{header.version} format {header.point_format.id}"
                        )
                    if not np.allclose(
                        header.scales, self.config.expected_scale, rtol=0, atol=0
                    ):
                        raise CopcPublisherError(
                            f"{path} has scales {header.scales}, expected "
                            f"{self.config.expected_scale}"
                        )
                    if not np.allclose(
                        header.offsets, self.config.expected_offset, rtol=0, atol=0
                    ):
                        raise CopcPublisherError(
                            f"{path} has offsets {header.offsets}, expected "
                            f"{self.config.expected_offset}"
                        )
                    missing = set(self.config.required_dimensions) - set(dimensions)
                    if missing:
                        raise CopcPublisherError(
                            f"{path} is missing required dimensions: {sorted(missing)}"
                        )
                    if schema is None:
                        schema = dimensions
                        point_format = header.point_format.id
                    elif dimensions != schema:
                        raise CopcPublisherError(
                            f"{path} does not share the first shard's dimension schema"
                        )
                    elif header.point_format.id != point_format:
                        raise CopcPublisherError(
                            f"{path} does not share the first shard's point format"
                        )
                    point_count += header.point_count
                    if self.config.validate_source_distribution:
                        for points in reader.chunk_iterator(2_000_000):
                            source_counts += np.bincount(
                                np.asarray(points.point_source_id),
                                minlength=65536,
                            )
            except CopcPublisherError:
                raise
            except Exception as exc:
                raise CopcPublisherError(
                    f"could not read input shard {path}: {exc}"
                ) from exc
        distribution = {
            int(index): int(source_counts[index])
            for index in np.flatnonzero(source_counts)
        }
        assert point_format is not None
        return _InputSummary(
            files,
            point_count,
            schema or (),
            point_format,
            distribution,
        )

    @staticmethod
    def _link_inputs(files: tuple[Path, ...], directory: Path) -> None:
        for index, source in enumerate(files):
            suffix = (
                "".join(source.suffixes[-2:])
                if source.name.endswith(".copc.laz")
                else source.suffix
            )
            target = directory / f"{index:06d}{suffix}"
            target.symlink_to(source)

    def _validate_output(
        self, path: Path, inputs: _InputSummary
    ) -> tuple[CopcStructure, dict[int, int]]:
        try:
            structure = validate_copc_structure(path, inputs.point_count)
        except (OSError, ValueError) as exc:
            raise CopcPublisherError(str(exc)) from exc
        if structure.point_format != inputs.point_format:
            raise CopcPublisherError(
                f"output point format is {structure.point_format}, "
                f"expected {inputs.point_format}"
            )
        if structure.scale != self.config.expected_scale:
            raise CopcPublisherError(
                f"output scales are {structure.scale}, expected {self.config.expected_scale}"
            )
        if structure.offset != self.config.expected_offset:
            raise CopcPublisherError(
                f"output offsets are {structure.offset}, expected {self.config.expected_offset}"
            )

        laspy = _laspy()
        output_counts = np.zeros(65536, dtype=np.int64)
        try:
            with laspy.open(path) as reader:
                dimensions = tuple(reader.header.point_format.dimension_names)
                missing = set(self.config.required_dimensions) - set(dimensions)
                if missing:
                    raise CopcPublisherError(
                        f"output is missing required dimensions: {sorted(missing)}"
                    )
                if dimensions != inputs.dimensions:
                    raise CopcPublisherError(
                        "output dimension schema differs from the input shards"
                    )
                if self.config.validate_source_distribution:
                    for points in reader.chunk_iterator(2_000_000):
                        output_counts += np.bincount(
                            np.asarray(points.point_source_id),
                            minlength=65536,
                        )
        except CopcPublisherError:
            raise
        except Exception as exc:
            raise CopcPublisherError(f"could not read output COPC: {exc}") from exc
        output_distribution = {
            int(index): int(output_counts[index])
            for index in np.flatnonzero(output_counts)
        }
        if (
            self.config.validate_source_distribution
            and output_distribution != inputs.source_distribution
        ):
            raise CopcPublisherError(
                "output PointSourceId distribution does not match the input shards"
            )
        return structure, output_distribution

    def publish(
        self,
        inputs: Path | Sequence[Path],
        output: Path,
        *,
        overwrite: bool = False,
    ) -> PublishResult:
        """Merge LAS/LAZ shards into a validated COPC.

        The final path is only replaced after the converter exits successfully
        and independent structural/schema validation passes.
        """

        output = Path(output)
        if not output.name.lower().endswith(".copc.laz"):
            raise ValueError("COPC output must end in .copc.laz")
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        files = self._discover(inputs)
        summary = self._summarize_inputs(files)
        executable = self.resolve_executable()
        version = self._check_version(executable)
        output.parent.mkdir(parents=True, exist_ok=True)

        temporary_output = output.with_name(
            f".{output.name}.{uuid.uuid4().hex}.partial.copc.laz"
        )
        scratch_parent = (
            Path(self.config.temp_dir) if self.config.temp_dir is not None else None
        )
        if scratch_parent is not None:
            scratch_parent.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(
                prefix="mapper-copc-", dir=scratch_parent
            ) as work_text:
                work = Path(work_text)
                links = work / "inputs"
                links.mkdir()
                self._link_inputs(files, links)
                scratch = work / "scratch"
                scratch.mkdir()
                command = [
                    str(executable),
                    str(links),
                    str(temporary_output),
                    "--memory-limit",
                    self.config.memory_limit,
                    "--temp-dir",
                    str(scratch),
                    "--temp-compression",
                    self.config.temp_compression,
                    "--node-storage",
                    self.config.node_storage,
                    "--progress",
                    self.config.progress,
                ]
                if self.config.threads is not None:
                    command.extend(["--threads", str(self.config.threads)])
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                )
                if completed.returncode:
                    raise CopcPublisherError(
                        f"copc_converter exited {completed.returncode}: "
                        f"{completed.stderr.strip() or completed.stdout.strip()}"
                    )
                if self.config.flatten_hierarchy:
                    flatten_copc_hierarchy(temporary_output)
                structure, source_distribution = self._validate_output(
                    temporary_output, summary
                )
                os.replace(temporary_output, output)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
        wall_seconds = time.monotonic() - started
        return PublishResult(
            output=output,
            point_count=summary.point_count,
            file_bytes=output.stat().st_size,
            sha256=_sha256(output),
            wall_seconds=wall_seconds,
            converter_version=version,
            input_files=files,
            source_distribution=source_distribution,
            structure=structure,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
