"""Disk-backed, confidence-aware voxel consolidation for canonical LAZ shards."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .copc import CopcPublisher, CopcPublisherError
from .las_staging import LasStagingConfig, write_laz_shard


@dataclass(frozen=True)
class VoxelConsolidationConfig:
    voxel_size: float = 0.02
    max_bucket_points: int = 1_000_000
    read_chunk_points: int = 1_000_000
    temp_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if self.max_bucket_points < 1 or self.read_chunk_points < 1:
            raise ValueError("point limits must be positive")


@dataclass(frozen=True)
class ConsolidationResult:
    output_dir: Path
    shards: tuple[Path, ...]
    points_in: int
    points_out: int
    contributor_total: int
    voxel_size: float
    bucket_count: int
    max_bucket_points: int


def _laspy():
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("voxel consolidation requires laspy and lazrs") from exc
    return laspy


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _bucket_ids(x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: int) -> np.ndarray:
    # SplitMix-inspired odd constants give stable spatial-key diffusion. uint64
    # overflow is intentional and deterministic.
    with np.errstate(over="ignore"):
        hashed = (
            x.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
            ^ y.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ z.astype(np.uint64) * np.uint64(0x165667B19E3779F9)
        )
        hashed ^= hashed >> np.uint64(30)
    return (hashed & np.uint64(mask)).astype(np.int64)


def consolidate_laz_shards(
    inputs: Path | Sequence[Path],
    output_dir: Path,
    *,
    config: VoxelConsolidationConfig | None = None,
) -> ConsolidationResult:
    """Consolidate cross-shard voxels with bounded RAM and disk spill.

    Every voxel is hashed to exactly one temporary bucket, so contributors from
    different reconstruction windows/submaps meet before selection. Within a
    voxel, the highest-confidence record wins XYZ, RGB and ``PointSourceId``;
    ``ContributorCount`` is the sum of all incoming contributor counts.

    A bucket is never loaded after it exceeds ``max_bucket_points``. The
    conservative 50% target normally leaves ample hash-skew headroom; an
    adversarial/skewed input fails explicitly instead of violating the memory
    contract.
    """

    config = config or VoxelConsolidationConfig()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    files = CopcPublisher._discover(inputs)
    laspy = _laspy()

    points_in = 0
    has_color: bool | None = None
    has_confidence: bool | None = None
    for path in files:
        with laspy.open(path) as reader:
            header = reader.header
            if str(header.version) != "1.4" or header.point_format.id not in {6, 7}:
                raise CopcPublisherError(f"{path} must be LAS 1.4 point format 6 or 7")
            dimensions = set(header.point_format.dimension_names)
            missing = {"ContributorCount"} - dimensions
            if missing:
                raise CopcPublisherError(
                    f"{path} is missing consolidation dimensions: {sorted(missing)}"
                )
            current_color = header.point_format.id == 7
            current_confidence = "Confidence" in dimensions
            if has_color is None:
                has_color = current_color
            elif has_color != current_color:
                raise CopcPublisherError("all consolidation shards must agree on RGB")
            if has_confidence is None:
                has_confidence = current_confidence
            elif has_confidence != current_confidence:
                raise CopcPublisherError(
                    "all consolidation shards must agree on Confidence"
                )
            if tuple(header.scales) != (0.001, 0.001, 0.001) or tuple(
                header.offsets
            ) != (0.0, 0.0, 0.0):
                raise CopcPublisherError(f"{path} is not in the canonical LAS grid")
            points_in += header.point_count

    voxel_ticks_float = config.voxel_size / 0.001
    voxel_ticks = round(voxel_ticks_float)
    if voxel_ticks < 1 or not math.isclose(
        voxel_ticks_float, voxel_ticks, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("voxel_size must be an exact multiple of 0.001 metres")

    # Target half the hard cap so normal statistical skew remains bounded.
    bucket_count = _next_power_of_two(
        max(1, math.ceil(points_in / max(1, config.max_bucket_points // 2)))
    )
    bucket_mask = bucket_count - 1
    fields = [
        ("X", "<i4"),
        ("Y", "<i4"),
        ("Z", "<i4"),
        ("SourceIndex", "<u2"),
        ("ContributorCount", "<u2"),
    ]
    if has_confidence:
        fields.append(("Confidence", "<f4"))
    if has_color:
        fields.extend([("Red", "<u2"), ("Green", "<u2"), ("Blue", "<u2")])
    record_dtype = np.dtype(fields, align=False)

    scratch_parent = Path(config.temp_dir) if config.temp_dir else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_dir.with_name(
        f".{output_dir.name}.consolidating-{os.getpid()}"
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="mapper-voxel-", dir=scratch_parent
        ) as scratch_text:
            scratch = Path(scratch_text)
            bucket_paths = [
                scratch / f"bucket-{index:06d}.bin" for index in range(bucket_count)
            ]
            bucket_counts = np.zeros(bucket_count, dtype=np.int64)

            for path in files:
                with laspy.open(path) as reader:
                    for points in reader.chunk_iterator(config.read_chunk_points):
                        count = len(points)
                        records = np.empty(count, dtype=record_dtype)
                        records["X"] = np.asarray(points.X)
                        records["Y"] = np.asarray(points.Y)
                        records["Z"] = np.asarray(points.Z)
                        records["SourceIndex"] = np.asarray(points.point_source_id)
                        if has_confidence:
                            records["Confidence"] = np.asarray(points["Confidence"])
                        records["ContributorCount"] = np.asarray(
                            points["ContributorCount"]
                        )
                        if has_color:
                            records["Red"] = np.asarray(points.red)
                            records["Green"] = np.asarray(points.green)
                            records["Blue"] = np.asarray(points.blue)
                        vx = np.floor_divide(records["X"], voxel_ticks)
                        vy = np.floor_divide(records["Y"], voxel_ticks)
                        vz = np.floor_divide(records["Z"], voxel_ticks)
                        ids = _bucket_ids(vx, vy, vz, bucket_mask)
                        order = np.argsort(ids, kind="quicksort")
                        sorted_ids = ids[order]
                        starts = np.r_[
                            0, 1 + np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1])
                        ]
                        stops = np.r_[starts[1:], len(sorted_ids)]
                        for start, stop in zip(starts, stops):
                            bucket = int(sorted_ids[start])
                            selected = records[order[start:stop]]
                            with bucket_paths[int(bucket)].open("ab") as handle:
                                selected.tofile(handle)
                            bucket_counts[int(bucket)] += len(selected)

            largest = int(bucket_counts.max(initial=0))
            if largest > config.max_bucket_points:
                raise CopcPublisherError(
                    "voxel hash bucket exceeded the memory contract: "
                    f"{largest:,} > {config.max_bucket_points:,} points; "
                    "lower max_bucket_points to increase partition count"
                )

            temporary_output.mkdir(parents=True)
            points_out = 0
            contributor_total = 0
            shard_paths = []
            staging = LasStagingConfig(
                chunk_points=config.read_chunk_points,
                include_confidence=bool(has_confidence),
                include_contributor_count=True,
                color_bits=16,
            )
            for bucket, count in enumerate(bucket_counts):
                if count == 0:
                    continue
                records = np.memmap(
                    bucket_paths[bucket],
                    dtype=record_dtype,
                    mode="r",
                    shape=(int(count),),
                )
                vx = np.floor_divide(records["X"], voxel_ticks)
                vy = np.floor_divide(records["Y"], voxel_ticks)
                vz = np.floor_divide(records["Z"], voxel_ticks)
                # Primary sort: voxel XYZ. Within a voxel use highest
                # confidence then lowest source; without confidence the lowest
                # source wins. Raw XYZ makes remaining ties deterministic.
                tie_breakers = (
                    (
                        records["Z"],
                        records["Y"],
                        records["X"],
                        records["SourceIndex"],
                        -records["Confidence"],
                        vz,
                        vy,
                        vx,
                    )
                    if has_confidence
                    else (
                        records["Z"],
                        records["Y"],
                        records["X"],
                        records["SourceIndex"],
                        vz,
                        vy,
                        vx,
                    )
                )
                order = np.lexsort(tie_breakers)
                ordered = records[order]
                ovx = np.floor_divide(ordered["X"], voxel_ticks)
                ovy = np.floor_divide(ordered["Y"], voxel_ticks)
                ovz = np.floor_divide(ordered["Z"], voxel_ticks)
                starts = np.r_[
                    0,
                    1
                    + np.flatnonzero(
                        (ovx[1:] != ovx[:-1])
                        | (ovy[1:] != ovy[:-1])
                        | (ovz[1:] != ovz[:-1])
                    ),
                ]
                contributor_sums = np.add.reduceat(
                    ordered["ContributorCount"].astype(np.uint64), starts
                )
                if contributor_sums.size and contributor_sums.max() > 65535:
                    raise CopcPublisherError(
                        "ContributorCount overflow after voxel consolidation"
                    )
                winners = ordered[starts]
                xyz = np.column_stack(
                    (winners["X"], winners["Y"], winners["Z"])
                ).astype(np.float64)
                xyz *= 0.001
                colors = (
                    np.column_stack((winners["Red"], winners["Green"], winners["Blue"]))
                    if has_color
                    else None
                )
                shard = temporary_output / f"voxel-{bucket:06d}.laz"
                write_laz_shard(
                    shard,
                    xyz,
                    colors,
                    winners["SourceIndex"],
                    confidence=(winners["Confidence"] if has_confidence else None),
                    contributor_count=contributor_sums.astype(np.uint16),
                    config=staging,
                )
                shard_paths.append(shard)
                points_out += len(winners)
                contributor_total += int(contributor_sums.sum())
                del records, ordered, order

        os.replace(temporary_output, output_dir)
    except Exception:
        if temporary_output.exists():
            for child in temporary_output.iterdir():
                child.unlink(missing_ok=True)
            temporary_output.rmdir()
        raise

    final_shards = tuple(output_dir / path.name for path in shard_paths)
    return ConsolidationResult(
        output_dir=output_dir,
        shards=final_shards,
        points_in=points_in,
        points_out=points_out,
        contributor_total=contributor_total,
        voxel_size=config.voxel_size,
        bucket_count=bucket_count,
        max_bucket_points=largest,
    )
