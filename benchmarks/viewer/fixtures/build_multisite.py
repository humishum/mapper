"""Build distant-site and same-site comparison packages from one COPC."""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import laspy
import numpy as np
from pyproj import Transformer

from src.publisher.copc_validation import inspect_copc, validate_copc_structure
from src.publisher.package import _column_contracts, _write_canonical_parquet
from viewer.backend.domain.geospatial import global_placement_from_local_bounds
from viewer.backend.domain.package import (
    Alignment,
    CaptureMetadata,
    CoordinateFrame,
    LayerDefault,
    Manifest,
    Producer,
    SourceRecord,
    SourcesDocument,
)
from viewer.backend.services.catalog import Catalog
from viewer.backend.services.package_validator import sha256_file
from viewer.backend.services.package_writer import (
    describe_artifact,
    write_json_sidecar,
    write_manifest,
)

FIXTURE_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


@dataclass(frozen=True)
class SiteDefinition:
    key: str
    longitude: float
    latitude: float
    ellipsoidal_height_m: float


DEFAULT_SITES = (
    SiteDefinition("site_a", -122.4194, 37.7749, 30.0),
    SiteDefinition("site_b", -115.1398, 36.1699, 620.0),
)


@dataclass(frozen=True)
class BuiltSite:
    definition: SiteDefinition
    package_root: Path
    artifact_id: str
    geometry_disposition: str
    transform_to_ecef: tuple[float, ...]


@dataclass(frozen=True)
class MultiSiteFixture:
    catalog_path: Path
    sites: tuple[BuiltSite, ...]

    def summary(self) -> dict[str, object]:
        return {
            "catalog_path": str(self.catalog_path),
            "sites": [
                {
                    "key": site.definition.key,
                    "artifact_id": site.artifact_id,
                    "package_root": str(site.package_root),
                    "geometry_disposition": site.geometry_disposition,
                    "origin_wgs84": [
                        site.definition.longitude,
                        site.definition.latitude,
                        site.definition.ellipsoidal_height_m,
                    ],
                    "transform_to_ecef": list(site.transform_to_ecef),
                }
                for site in self.sites
            ],
        }


def _enu_to_ecef(site: SiteDefinition) -> list[float]:
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = to_ecef.transform(
        site.longitude, site.latitude, site.ellipsoidal_height_m
    )
    longitude = math.radians(site.longitude)
    latitude = math.radians(site.latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    return [
        -sin_lon,
        -sin_lat * cos_lon,
        cos_lat * cos_lon,
        float(x),
        cos_lon,
        -sin_lat * sin_lon,
        cos_lat * sin_lon,
        float(y),
        0.0,
        cos_lat,
        sin_lat,
        float(z),
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _adopt(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=False)
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError as exc:
        if exc.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise
    shutil.copy2(source, target)
    return "copied"


@dataclass(frozen=True)
class CopcCapabilities:
    required_dimensions: tuple[str, ...]
    source_distribution: tuple[tuple[int, int], ...]


def inspect_copc_capabilities(
    source: Path, expected_point_count: int
) -> CopcCapabilities:
    """Read real COPC dimensions and its complete PointSourceId distribution."""
    with laspy.open(source) as reader:
        if reader.header.point_count != expected_point_count:
            raise ValueError(
                "LAS header point count does not match the structural COPC count"
            )
        names = tuple(str(name) for name in reader.header.point_format.dimension_names)
        # laspy exposes standard LAS dimensions in snake_case
        # (`point_source_id`) while extra bytes retain their declared names.
        # Normalize separators before matching the canonical browser contract.
        available = {name.replace("_", "").casefold(): name for name in names}
        point_source_name = available.get("pointsourceid")
        contributor_name = available.get("contributorcount")
        if point_source_name is None:
            raise ValueError("fixture COPC requires the canonical PointSourceId dimension")
        if contributor_name is None:
            raise ValueError(
                "fixture COPC requires the canonical ContributorCount dimension"
            )
        required = ["X", "Y", "Z"]
        if all(channel in available for channel in ("red", "green", "blue")):
            required.extend(["Red", "Green", "Blue"])
        required.extend(["PointSourceId", "ContributorCount"])
        if "confidence" in available:
            required.append("Confidence")

        distribution: dict[int, int] = {}
        observed = 0
        for points in reader.chunk_iterator(1_000_000):
            source_ids = np.asarray(points[point_source_name])
            contributors = np.asarray(points[contributor_name])
            if len(contributors) and int(contributors.min()) < 1:
                raise ValueError("ContributorCount values must be at least one")
            indexes, counts = np.unique(source_ids, return_counts=True)
            for index, count in zip(indexes, counts, strict=True):
                source_index = int(index)
                if not 0 <= source_index <= 65535:
                    raise ValueError("PointSourceId values must fit the uint16 contract")
                distribution[source_index] = distribution.get(source_index, 0) + int(count)
            observed += len(points)
    if observed != expected_point_count:
        raise ValueError(
            f"read {observed} provenance values for {expected_point_count} points"
        )
    if expected_point_count and not distribution:
        raise ValueError("non-empty fixture COPC has no PointSourceId distribution")
    return CopcCapabilities(
        required_dimensions=tuple(required),
        source_distribution=tuple(sorted(distribution.items())),
    )


def _build_site(
    source_copc: Path,
    output_root: Path,
    site: SiteDefinition,
    *,
    point_count: int,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    capabilities: CopcCapabilities,
    source_checksum: str,
) -> BuiltSite:
    package_root = output_root / site.key
    if package_root.exists():
        raise FileExistsError(f"fixture package already exists: {package_root}")

    geometry_path = package_root / "geometry" / "points.copc.laz"
    disposition = _adopt(source_copc, geometry_path)
    transform = _enu_to_ecef(site)
    placement = global_placement_from_local_bounds(
        bounds_min, bounds_max, transform
    )
    capture_id = f"capture_phase2_{site.key}"
    run_id = f"run_phase2_{site.key}"
    artifact_id = f"artifact_phase2_{site.key}"

    sources = SourcesDocument(
        provenance_dimension="PointSourceId",
        granularity="batch",
        sources=[
            SourceRecord(
                source_index=source_index,
                kind="batch",
                capture_id=capture_id,
                run_id=run_id,
                name=f"Phase 2 synthetic {site.key} source {source_index}",
                point_count=source_point_count,
                metadata={
                    "synthetic_fixture": True,
                    "source_copc_sha256": source_checksum,
                },
            )
            for source_index, source_point_count in capabilities.source_distribution
        ],
    )
    write_json_sidecar(package_root, "metadata/sources.json", sources)
    pose_path = package_root / "cameras" / "poses.parquet"
    center = [
        (low + high) / 2
        for low, high in zip(bounds_min, bounds_max, strict=True)
    ]
    _write_canonical_parquet(
        pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0, 2.0],
                "frame_index": [0, 1, 2],
                "tx": [center[0] - 1.0, center[0], center[0] + 1.0],
                "ty": [center[1], center[1], center[1]],
                "tz": [center[2] + 1.0, center[2] + 1.0, center[2] + 1.0],
                "qx": [0.0, 0.0, 0.0],
                "qy": [0.0, 0.0, 0.0],
                "qz": [0.0, 0.0, 0.0],
                "qw": [1.0, 1.0, 1.0],
                "fx": [500.0, 500.0, 500.0],
                "fy": [500.0, 500.0, 500.0],
                "cx": [320.0, 320.0, 320.0],
                "cy": [240.0, 240.0, 240.0],
            }
        ),
        pose_path,
        "poses",
    )

    point_artifact = describe_artifact(
        package_root,
        "geometry/points.copc.laz",
        representation_id=f"representation_phase2_{site.key}_points",
        kind="points",
        format="copc/laz",
        media_type="application/vnd.laszip",
        frame="artifact_local",
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        point_count=point_count,
        required_dimensions=list(capabilities.required_dimensions),
        metadata={
            "fixture_source_sha256": source_checksum,
        },
    )
    sources_artifact = describe_artifact(
        package_root,
        "metadata/sources.json",
        representation_id=f"representation_phase2_{site.key}_sources",
        kind="sources",
        format="json",
        media_type="application/json",
    )
    poses_artifact = describe_artifact(
        package_root,
        "cameras/poses.parquet",
        representation_id=f"representation_phase2_{site.key}_poses",
        kind="poses",
        format="parquet",
        media_type="application/vnd.apache.parquet",
        frame="artifact_local",
        columns=_column_contracts("poses"),
    )
    manifest = Manifest(
        run_id=run_id,
        capture_id=capture_id,
        artifact_id=artifact_id,
        created_at=FIXTURE_CREATED_AT,
        capture=CaptureMetadata(
            device="deterministic synthetic fixture",
        ),
        producer=Producer(
            model_name="phase2-synthetic-multisite",
            model_version="1",
            model_config={"source_copc_sha256": source_checksum},
            capabilities={"deterministic": True},
            git_status="unknown",
            adapter_name="existing-copc-fixture",
            adapter_version="1",
            publisher_name="phase2-fixture-builder",
            publisher_version="1",
        ),
        coordinate_frame=CoordinateFrame(
            name="artifact_local",
            units="metre",
            axis_order=["east", "north", "up"],
            handedness="right",
            origin_wgs84=placement.origin_wgs84,
            transform_to_ecef=transform,
        ),
        alignment=Alignment(
            status="aligned",
            method="deterministic_synthetic_anchor",
            model_to_artifact_local=IDENTITY,
            scale=1.0,
            inlier_count=0,
            horizontal_rmse_m=0.0,
            vertical_rmse_m=0.0,
            gravity_constrained=True,
        ),
        footprint_wgs84=placement.footprint_wgs84,
        artifacts=[point_artifact, poses_artifact, sources_artifact],
        layer_default=LayerDefault(
            visible=True,
            point_budget=2_000_000,
            point_size=2.0,
            color_dimension=(
                "RGB"
                if {"Red", "Green", "Blue"}.issubset(
                    capabilities.required_dimensions
                )
                else None
            ),
        ),
    )
    write_manifest(package_root, manifest)
    return BuiltSite(
        definition=site,
        package_root=package_root.resolve(),
        artifact_id=artifact_id,
        geometry_disposition=disposition,
        transform_to_ecef=tuple(transform),
    )


def build_multisite_fixture(
    source_copc: str | Path,
    output_root: str | Path,
    *,
    catalog_path: str | Path | None = None,
    sites: tuple[SiteDefinition, SiteDefinition] = DEFAULT_SITES,
) -> MultiSiteFixture:
    """Build two distant sites plus a same-site comparison package."""
    source = Path(source_copc).resolve(strict=True)
    if not source.is_file():
        raise ValueError("source COPC must be a regular file")
    if len(sites) != 2 or sites[0].key == sites[1].key:
        raise ValueError("exactly two sites with distinct keys are required")
    comparison_site = SiteDefinition(
        key=f"{sites[0].key}_comparison",
        longitude=sites[0].longitude,
        latitude=sites[0].latitude,
        ellipsoidal_height_m=sites[0].ellipsoidal_height_m,
    )

    structure = inspect_copc(source)
    validate_copc_structure(source, structure.point_count)
    capabilities = inspect_copc_capabilities(source, structure.point_count)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination_catalog = (
        Path(catalog_path).resolve()
        if catalog_path is not None
        else root / "catalog.sqlite3"
    )
    if destination_catalog == source:
        raise ValueError("catalog path must not overwrite the source COPC")
    all_sites = (*sites, comparison_site)
    for site in all_sites:
        try:
            destination_catalog.relative_to(root / site.key)
        except ValueError:
            continue
        raise ValueError("catalog path must be outside immutable package roots")
    checksum = sha256_file(source)
    built = tuple(
        _build_site(
            source,
            root,
            site,
            point_count=structure.point_count,
            bounds_min=structure.bounds_min,
            bounds_max=structure.bounds_max,
            capabilities=capabilities,
            source_checksum=checksum,
        )
        for site in all_sites
    )
    catalog = Catalog(destination_catalog)
    for site in built:
        catalog.register_package(site.package_root)
    return MultiSiteFixture(
        catalog_path=destination_catalog,
        sites=built,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_copc", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()
    fixture = build_multisite_fixture(
        arguments.source_copc,
        arguments.output_root,
        catalog_path=arguments.catalog,
    )
    summary = json.dumps(fixture.summary(), indent=2, sort_keys=True) + "\n"
    if arguments.summary is None:
        print(summary, end="")
    else:
        arguments.summary.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
