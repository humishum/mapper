"""Bounded-memory staging of model point arrays into canonical LAZ shards."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass(frozen=True)
class LasStagingConfig:
    """Schema and memory controls for canonical publisher input shards."""

    chunk_points: int = 1_000_000
    scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    include_confidence: bool | None = None
    include_contributor_count: bool = True
    color_bits: int = 8

    def __post_init__(self) -> None:
        if self.chunk_points < 1:
            raise ValueError("chunk_points must be positive")
        if self.color_bits not in {8, 16}:
            raise ValueError("color_bits must be 8 or 16")
        if any(value <= 0 for value in self.scale):
            raise ValueError("LAS scales must be positive")


@dataclass(frozen=True)
class LasShardResult:
    path: Path
    point_count: int
    file_bytes: int
    sha256: str
    source_indices: tuple[int, ...]
    dimensions: tuple[str, ...]


def _laspy():
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "LAS staging requires the project's laspy and lazrs dependencies"
        ) from exc
    return laspy


def _as_vector(
    value: Union[int, np.ndarray],
    point_count: int,
    name: str,
    dtype: np.dtype,
) -> Union[int, np.ndarray]:
    if np.isscalar(value):
        scalar = int(value)
        if not 0 <= scalar <= np.iinfo(dtype).max:
            raise ValueError(f"{name} scalar is outside {dtype} range")
        return scalar
    array = np.asarray(value)
    if array.shape != (point_count,):
        raise ValueError(f"{name} must have shape ({point_count},), got {array.shape}")
    if array.size and (np.min(array) < 0 or np.max(array) > np.iinfo(dtype).max):
        raise ValueError(f"{name} contains values outside {dtype} range")
    return array.astype(dtype, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_laz_shard(
    output: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    source_index: Union[int, np.ndarray],
    *,
    confidence: np.ndarray | None = None,
    contributor_count: Union[int, np.ndarray] = 1,
    config: LasStagingConfig | None = None,
    overwrite: bool = False,
) -> LasShardResult:
    """Write one canonical point-format-7 LAZ shard without a merged copy.

    ``source_index`` and ``contributor_count`` accept scalars, so callers that
    publish one source unit per shard do not need to allocate ``np.repeat``
    arrays.  Array values are supported for migration from already-merged data.
    """

    config = config or LasStagingConfig()
    output = Path(output)
    if output.suffix.lower() != ".laz":
        raise ValueError("canonical staging output must end in .laz")
    if output.exists() and not overwrite:
        raise FileExistsError(output)

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    point_count = len(points)
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite coordinates")
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != (point_count, 3):
            raise ValueError(
                f"colors must have shape ({point_count}, 3), got {colors.shape}"
            )
        max_color = (1 << config.color_bits) - 1
        if colors.size and (np.min(colors) < 0 or np.max(colors) > max_color):
            raise ValueError(f"colors must be in the {config.color_bits}-bit range")

    source_values = _as_vector(
        source_index, point_count, "source_index", np.dtype(np.uint16)
    )
    contributor_values = _as_vector(
        contributor_count,
        point_count,
        "contributor_count",
        np.dtype(np.uint16),
    )

    include_confidence = (
        confidence is not None
        if config.include_confidence is None
        else config.include_confidence
    )
    confidence_values = None
    if include_confidence:
        if confidence is None:
            raise ValueError("confidence is required by this staging schema")
        confidence_values = np.asarray(confidence)
        if confidence_values.shape != (point_count,):
            raise ValueError(
                f"confidence must have shape ({point_count},), "
                f"got {confidence_values.shape}"
            )
        if not np.isfinite(confidence_values).all():
            raise ValueError("confidence contains non-finite values")
    elif confidence is not None:
        raise ValueError("confidence was supplied but include_confidence is false")

    laspy = _laspy()
    header = laspy.LasHeader(point_format=7 if colors is not None else 6, version="1.4")
    header.scales = np.asarray(config.scale, dtype=np.float64)
    header.offsets = np.asarray(config.offset, dtype=np.float64)
    if include_confidence:
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="Confidence",
                type=np.float32,
                description="model confidence",
            )
        )
    if config.include_contributor_count:
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="ContributorCount",
                type=np.uint16,
                description="voxel contributor count",
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
    )
    try:
        with laspy.open(temporary, mode="w", header=header, do_compress=True) as writer:
            for start in range(0, point_count, config.chunk_points):
                stop = min(start + config.chunk_points, point_count)
                record = laspy.ScaleAwarePointRecord.zeros(stop - start, header=header)
                record.x = points[start:stop, 0]
                record.y = points[start:stop, 1]
                record.z = points[start:stop, 2]
                if colors is not None:
                    color_chunk = colors[start:stop].astype(np.uint16, copy=False)
                    if config.color_bits == 8:
                        color_chunk = color_chunk * np.uint16(257)
                    record.red = color_chunk[:, 0]
                    record.green = color_chunk[:, 1]
                    record.blue = color_chunk[:, 2]
                if np.isscalar(source_values):
                    record.point_source_id[:] = source_values
                else:
                    record.point_source_id = source_values[start:stop]
                if include_confidence:
                    record["Confidence"] = confidence_values[start:stop].astype(
                        np.float32, copy=False
                    )
                if config.include_contributor_count:
                    if np.isscalar(contributor_values):
                        record["ContributorCount"][:] = contributor_values
                    else:
                        record["ContributorCount"] = contributor_values[start:stop]
                writer.write_points(record)
        if overwrite:
            os.replace(temporary, output)
        else:
            # The existence check above makes replace safe while retaining atomic
            # publication. A concurrent creator wins and is never overwritten.
            try:
                os.link(temporary, output)
            except FileExistsError:
                raise
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if np.isscalar(source_values):
        source_indices = (int(source_values),)
    else:
        source_indices = tuple(int(value) for value in np.unique(source_values))
    return LasShardResult(
        path=output,
        point_count=point_count,
        file_bytes=output.stat().st_size,
        sha256=_sha256(output),
        source_indices=source_indices,
        dimensions=tuple(header.point_format.dimension_names),
    )
