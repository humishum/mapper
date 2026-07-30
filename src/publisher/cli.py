"""Command-line entry points for staging and COPC publication."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .copc import CopcPublisher, CopcPublisherConfig
from .legacy_ply import (
    LegacyPlyStagingConfig,
    source_counts_from_window_dir,
    stage_legacy_binary_ply,
)
from .migration import LegacyMigrationResult, migrate_legacy_ply_to_copc
from .onboarding import ExistingCopcPackageResult, ExistingCopcPackager


def _json_result(result):
    if isinstance(result, ExistingCopcPackageResult):
        return {
            "package_root": str(result.package_root),
            "artifact_id": result.manifest.artifact_id,
            "alignment_status": result.manifest.alignment.status,
            "alignment_rejection_reason": result.manifest.alignment.rejection_reason,
            "input_sha256": result.input_sha256,
            "geometry_sha256": result.geometry_sha256,
            "geometry_disposition": result.geometry_disposition,
            "incompatibilities": result.incompatibilities,
            "catalog_record": result.catalog_record,
        }
    if isinstance(result, LegacyMigrationResult):
        publish = result.publish
        return {
            "output": str(result.output),
            "input_points": result.input_points,
            "output_points": result.output_points,
            "source_count": result.source_count,
            "has_color": result.has_color,
            "has_confidence": result.has_confidence,
            "voxel_size": result.voxel_size,
            "publication": {
                "file_bytes": publish.file_bytes,
                "sha256": publish.sha256,
                "wall_seconds": publish.wall_seconds,
                "converter_version": publish.converter_version,
                "source_distribution": publish.source_distribution,
                "structure": asdict(publish.structure),
            },
        }
    return asdict(result)


def _counts(path: Path | None):
    if path is None:
        return None
    value = json.loads(path.read_text())
    if isinstance(value, dict) and "provenance" in value:
        value = [item["point_count"] for item in value["provenance"]["windows"]]
    if not isinstance(value, list):
        raise ValueError("source counts JSON must be a list or Phase 0 report")
    return [int(item) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser(prog="mapper-publisher")
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage-ply")
    stage.add_argument("input", type=Path)
    stage.add_argument("output_dir", type=Path)
    stage.add_argument("--source-counts-json", type=Path)
    stage.add_argument("--source-window-dir", type=Path)
    stage.add_argument("--max-points-per-shard", type=int, default=1_000_000)

    publish = commands.add_parser("publish")
    publish.add_argument("input", type=Path)
    publish.add_argument("output", type=Path)
    publish.add_argument("--memory-limit", default="4G")
    publish.add_argument("--threads", type=int)
    publish.add_argument("--temp-dir", type=Path)

    migrate = commands.add_parser("migrate-ply")
    migrate.add_argument("input", type=Path)
    migrate.add_argument("output", type=Path)
    migrate.add_argument("--source-counts-json", type=Path)
    migrate.add_argument("--source-window-dir", type=Path)
    migrate.add_argument("--max-points-per-shard", type=int, default=1_000_000)
    migrate.add_argument("--memory-limit", default="4G")
    migrate.add_argument("--threads", type=int)
    migrate.add_argument("--temp-dir", type=Path)
    migrate.add_argument("--voxel-size", type=float)

    existing = commands.add_parser(
        "package-existing-copc",
        help="wrap an existing COPC and legacy sidecars in a registered package",
    )
    existing.add_argument("existing_copc", type=Path)
    existing.add_argument("legacy_reconstruction_dir", type=Path)
    existing.add_argument("package_root", type=Path)
    existing.add_argument("--catalog", type=Path, required=True)
    existing.add_argument("--source-video", type=Path)
    existing.add_argument("--memory-limit", default="4G")
    existing.add_argument("--threads", type=int)
    existing.add_argument("--temp-dir", type=Path)

    args = parser.parse_args()
    if args.command in {"stage-ply", "migrate-ply"}:
        if args.source_counts_json and args.source_window_dir:
            parser.error(
                "--source-counts-json and --source-window-dir are mutually exclusive"
            )
        source_counts = (
            source_counts_from_window_dir(args.source_window_dir)
            if args.source_window_dir
            else _counts(args.source_counts_json)
        )
        if args.command == "stage-ply":
            result = stage_legacy_binary_ply(
                args.input,
                args.output_dir,
                source_point_counts=source_counts,
                config=LegacyPlyStagingConfig(
                    max_points_per_shard=args.max_points_per_shard
                ),
            )
        else:
            result = migrate_legacy_ply_to_copc(
                args.input,
                args.output,
                source_point_counts=source_counts,
                max_points_per_shard=args.max_points_per_shard,
                voxel_size=args.voxel_size,
                publisher_config=CopcPublisherConfig(
                    memory_limit=args.memory_limit,
                    threads=args.threads,
                    temp_dir=args.temp_dir,
                ),
            )
    elif args.command == "publish":
        result = CopcPublisher(
            CopcPublisherConfig(
                memory_limit=args.memory_limit,
                threads=args.threads,
                temp_dir=args.temp_dir,
            )
        ).publish(args.input, args.output)
    else:
        result = ExistingCopcPackager(
            CopcPublisher(
                CopcPublisherConfig(
                    memory_limit=args.memory_limit,
                    threads=args.threads,
                    temp_dir=args.temp_dir,
                )
            )
        ).package(
            args.existing_copc,
            args.legacy_reconstruction_dir,
            args.package_root,
            args.catalog,
            source_video=args.source_video,
        )
    print(json.dumps(_json_result(result), default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
