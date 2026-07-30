"""Streaming/memory-mapped migration from legacy binary PLY to LAZ shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .las_staging import LasStagingConfig, write_laz_shard


@dataclass(frozen=True)
class LegacyPlyStagingConfig:
    max_points_per_shard: int = 1_000_000
    color_bits: int = 8

    def __post_init__(self) -> None:
        if self.max_points_per_shard < 1:
            raise ValueError("max_points_per_shard must be positive")


@dataclass(frozen=True)
class LegacyPlyStagingResult:
    output_dir: Path
    shards: tuple[Path, ...]
    point_count: int
    source_count: int
    has_color: bool
    has_confidence: bool


def source_counts_from_window_dir(window_dir: Path) -> tuple[int, ...]:
    """Read ordered legacy window point counts for merged-PLY provenance."""

    window_dir = Path(window_dir)
    metadata_files = sorted(window_dir.glob("window_*/metadata.json"))
    if not metadata_files:
        raise ValueError(f"no window metadata found below {window_dir}")
    counts = []
    for expected, path in enumerate(metadata_files):
        metadata = json.loads(path.read_text())
        window_id = int(
            metadata.get("window_metadata", {}).get(
                "window_id", metadata.get("window_id", expected)
            )
        )
        if window_id != expected:
            raise ValueError(
                f"window metadata is not contiguous: expected {expected}, "
                f"got {window_id} at {path}"
            )
        count = int(metadata["point_count"])
        if count < 0:
            raise ValueError(f"negative point_count in {path}")
        counts.append(count)
    return tuple(counts)


def stage_legacy_binary_ply(
    input_ply: Path,
    output_dir: Path,
    *,
    source_point_counts: Sequence[int] | None = None,
    config: LegacyPlyStagingConfig | None = None,
) -> LegacyPlyStagingResult:
    """Memory-map a legacy binary PLY and emit bounded canonical LAZ shards.

    ``source_point_counts`` restores provenance for old merged PLYs whose points
    are an ordered concatenation of reconstruction windows. If omitted, the
    whole PLY is source zero. A source may span multiple bounded shards without
    changing its ``PointSourceId``.
    """

    config = config or LegacyPlyStagingConfig()
    input_ply = Path(input_ply)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("legacy PLY migration requires plyfile") from exc

    ply = PlyData.read(str(input_ply), mmap="c")
    if ply.text:
        raise ValueError(
            "ASCII PLY is not supported by the bounded-memory migration path"
        )
    if "vertex" not in ply:
        raise ValueError("PLY has no vertex element")
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"PLY is missing coordinate fields: {sorted(missing)}")
    has_color = {"red", "green", "blue"} <= names
    partial_color = bool({"red", "green", "blue"} & names) and not has_color
    if partial_color:
        raise ValueError("PLY must contain all three RGB fields or none")
    has_confidence = "confidence" in names or "Confidence" in names
    confidence_name = "confidence" if "confidence" in names else "Confidence"
    point_count = len(vertices)

    if source_point_counts is None:
        counts = (point_count,)
    else:
        counts = tuple(int(value) for value in source_point_counts)
        if not counts or any(value < 0 for value in counts):
            raise ValueError("source_point_counts must be non-negative")
        if len(counts) > 65536:
            raise ValueError("source count exceeds uint16 PointSourceId capacity")
        if sum(counts) != point_count:
            raise ValueError(
                f"source counts sum to {sum(counts):,}, "
                f"but PLY has {point_count:,} points"
            )

    output_dir.mkdir(parents=True)
    shards = []
    cursor = 0
    try:
        for source_index, source_count in enumerate(counts):
            source_stop = cursor + source_count
            part = 0
            while cursor < source_stop:
                stop = min(cursor + config.max_points_per_shard, source_stop)
                view = vertices[cursor:stop]
                points = np.column_stack((view["x"], view["y"], view["z"]))
                colors = (
                    np.column_stack((view["red"], view["green"], view["blue"]))
                    if has_color
                    else None
                )
                confidence = (
                    np.asarray(view[confidence_name]) if has_confidence else None
                )
                shard = output_dir / (f"source-{source_index:05d}-part-{part:05d}.laz")
                write_laz_shard(
                    shard,
                    points,
                    colors,
                    source_index,
                    confidence=confidence,
                    contributor_count=1,
                    config=LasStagingConfig(
                        chunk_points=config.max_points_per_shard,
                        include_confidence=has_confidence,
                        include_contributor_count=True,
                        color_bits=config.color_bits,
                    ),
                )
                shards.append(shard)
                cursor = stop
                part += 1
    except Exception:
        for shard in shards:
            shard.unlink(missing_ok=True)
        output_dir.rmdir()
        raise

    return LegacyPlyStagingResult(
        output_dir=output_dir,
        shards=tuple(shards),
        point_count=point_count,
        source_count=len(counts),
        has_color=has_color,
        has_confidence=has_confidence,
    )
