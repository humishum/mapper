"""One-shot legacy PLY to validated COPC migration workflow."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .consolidation import VoxelConsolidationConfig, consolidate_laz_shards
from .copc import CopcPublisher, CopcPublisherConfig, PublishResult
from .legacy_ply import LegacyPlyStagingConfig, stage_legacy_binary_ply


@dataclass(frozen=True)
class LegacyMigrationResult:
    output: Path
    input_points: int
    output_points: int
    source_count: int
    has_color: bool
    has_confidence: bool
    voxel_size: float | None
    publish: PublishResult


def migrate_legacy_ply_to_copc(
    input_ply: Path,
    output: Path,
    *,
    source_point_counts: Sequence[int] | None = None,
    max_points_per_shard: int = 1_000_000,
    voxel_size: float | None = None,
    publisher_config: CopcPublisherConfig | None = None,
) -> LegacyMigrationResult:
    """Stage, optionally consolidate, publish, validate, and clean up."""

    publisher_config = publisher_config or CopcPublisherConfig()
    scratch_parent = (
        Path(publisher_config.temp_dir)
        if publisher_config.temp_dir is not None
        else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="mapper-ply-migration-", dir=scratch_parent
    ) as work_text:
        work = Path(work_text)
        staged = stage_legacy_binary_ply(
            input_ply,
            work / "staged",
            source_point_counts=source_point_counts,
            config=LegacyPlyStagingConfig(max_points_per_shard=max_points_per_shard),
        )
        publish_inputs: Path = staged.output_dir
        output_points = staged.point_count
        if voxel_size is not None:
            consolidated = consolidate_laz_shards(
                staged.output_dir,
                work / "consolidated",
                config=VoxelConsolidationConfig(
                    voxel_size=voxel_size,
                    max_bucket_points=max_points_per_shard,
                    read_chunk_points=max_points_per_shard,
                    temp_dir=work,
                ),
            )
            publish_inputs = consolidated.output_dir
            output_points = consolidated.points_out
        published = CopcPublisher(publisher_config).publish(publish_inputs, output)

    return LegacyMigrationResult(
        output=Path(output),
        input_points=staged.point_count,
        output_points=output_points,
        source_count=staged.source_count,
        has_color=staged.has_color,
        has_confidence=staged.has_confidence,
        voxel_size=voxel_size,
        publish=published,
    )
