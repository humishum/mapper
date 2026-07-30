from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import laspy
import numpy as np
import pytest

import benchmarks.viewer.fixtures.build_multisite as fixture_module
from benchmarks.viewer.fixtures import (
    CopcCapabilities,
    build_multisite_fixture,
    inspect_copc_capabilities,
)
from viewer.backend.services.catalog import Catalog
from viewer.backend.services.package_validator import PackageValidator, sha256_file


def _write_structural_copc(path: Path) -> None:
    """Write the smallest hierarchy-complete COPC accepted by the structural parser."""
    header = bytearray(375)
    header[:4] = b"LASF"
    header[24:26] = bytes((1, 4))
    header[104] = 6
    struct.pack_into("<H", header, 105, 30)
    struct.pack_into("<Q", header, 247, 1)
    struct.pack_into("<3d", header, 131, 0.01, 0.01, 0.01)
    struct.pack_into("<3d", header, 155, 0.0, 0.0, 0.0)
    struct.pack_into("<6d", header, 179, 10.0, -10.0, 5.0, -5.0, 2.0, -1.0)

    vlr = bytearray(54)
    vlr[2:6] = b"copc"
    struct.pack_into("<H", vlr, 18, 1)
    struct.pack_into("<H", vlr, 20, 160)

    hierarchy_offset = len(header) + len(vlr) + 160
    info = bytearray(160)
    struct.pack_into("<2Q", info, 40, hierarchy_offset, 32)
    root = struct.pack("<4iQii", 0, 0, 0, 0, hierarchy_offset + 32, 0, 1)
    path.write_bytes(header + vlr + info + root)


def _translation(transform: tuple[float, ...]) -> tuple[float, float, float]:
    return transform[3], transform[7], transform[11]


def _stub_capabilities(monkeypatch) -> CopcCapabilities:
    capabilities = CopcCapabilities(
        required_dimensions=(
            "X",
            "Y",
            "Z",
            "Red",
            "Green",
            "Blue",
            "PointSourceId",
            "ContributorCount",
            "Confidence",
        ),
        source_distribution=((4, 1),),
    )
    monkeypatch.setattr(
        fixture_module,
        "inspect_copc_capabilities",
        lambda _path, _count: capabilities,
    )
    return capabilities


def test_multisite_builder_registers_contained_distant_aligned_packages(
    tmp_path, monkeypatch
):
    capabilities = _stub_capabilities(monkeypatch)
    source = tmp_path / "source.copc.laz"
    _write_structural_copc(source)
    original_checksum = sha256_file(source)

    fixture = build_multisite_fixture(source, tmp_path / "fixture")

    assert fixture.catalog_path == (tmp_path / "fixture" / "catalog.sqlite3")
    assert sha256_file(source) == original_checksum
    assert math.dist(
        _translation(fixture.sites[0].transform_to_ecef),
        _translation(fixture.sites[1].transform_to_ecef),
    ) > 500_000
    assert math.dist(
        _translation(fixture.sites[0].transform_to_ecef),
        _translation(fixture.sites[2].transform_to_ecef),
    ) == 0

    catalog = Catalog(fixture.catalog_path)
    records = catalog.query_artifacts()
    assert {record["artifact_id"] for record in records} == {
        "artifact_phase2_site_a",
        "artifact_phase2_site_b",
        "artifact_phase2_site_a_comparison",
    }
    assert all(record["alignment_status"] == "aligned" for record in records)
    assert all(record["footprint"] is not None for record in records)

    validator = PackageValidator()
    for site in fixture.sites:
        package = validator.validate(site.package_root)
        assert package.manifest.coordinate_frame.transform_to_ecef is not None
        assert package.manifest.alignment.status == "aligned"
        assert package.sources is not None
        point_artifact = next(
            artifact for artifact in package.manifest.artifacts
            if artifact.kind == "points"
        )
        assert tuple(point_artifact.required_dimensions) == (
            capabilities.required_dimensions
        )
        assert {"source", "confidence"}.issubset(
            {
                "source" if name == "PointSourceId" else name.casefold()
                for name in point_artifact.required_dimensions
            }
        )
        assert [
            (record.source_index, record.point_count)
            for record in package.sources.sources
        ] == [(4, 1)]
        for artifact in package.manifest.artifacts:
            resolved = (site.package_root / artifact.path).resolve()
            resolved.relative_to(site.package_root.resolve())
            assert resolved.is_file()
        geometry = site.package_root / "geometry" / "points.copc.laz"
        assert not geometry.is_symlink()
        assert geometry.read_bytes() == source.read_bytes()

    site_a = catalog.get_artifact("artifact_phase2_site_a")
    comparison = catalog.get_artifact("artifact_phase2_site_a_comparison")
    assert site_a["artifact_id"] != comparison["artifact_id"]
    assert site_a["footprint"] == comparison["footprint"]


def test_multisite_manifests_and_source_metadata_are_reproducible(
    tmp_path, monkeypatch
):
    _stub_capabilities(monkeypatch)
    source = tmp_path / "source.copc.laz"
    _write_structural_copc(source)

    first = build_multisite_fixture(source, tmp_path / "first")
    second = build_multisite_fixture(source, tmp_path / "second")

    for first_site, second_site in zip(first.sites, second.sites, strict=True):
        assert first_site.artifact_id == second_site.artifact_id
        assert (
            first_site.package_root / "manifest.json"
        ).read_bytes() == (
            second_site.package_root / "manifest.json"
        ).read_bytes()
        first_sources = first_site.package_root / "metadata" / "sources.json"
        second_sources = second_site.package_root / "metadata" / "sources.json"
        assert first_sources.read_bytes() == second_sources.read_bytes()
        source_document = json.loads(first_sources.read_text(encoding="utf-8"))
        assert source_document["sources"][0]["metadata"] == {
            "synthetic_fixture": True,
            "source_copc_sha256": sha256_file(source),
        }


def test_real_las_dimensions_and_provenance_distribution_are_declared(tmp_path):
    path = tmp_path / "provenance.las"
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="PointSourceId", type=np.uint16)
    )
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="ContributorCount", type=np.uint16)
    )
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="Confidence", type=np.float32)
    )
    points = laspy.LasData(header)
    points.x = [0.0, 1.0, 2.0, 3.0]
    points.y = [0.0, 0.0, 0.0, 0.0]
    points.z = [0.0, 0.0, 0.0, 0.0]
    points.red = [1, 2, 3, 4]
    points.green = [1, 2, 3, 4]
    points.blue = [1, 2, 3, 4]
    points["PointSourceId"] = [7, 2, 7, 7]
    points["ContributorCount"] = [1, 2, 1, 3]
    points["Confidence"] = [0.5, 0.6, 0.7, 0.8]
    points.write(path)

    capabilities = inspect_copc_capabilities(path, 4)

    assert capabilities.required_dimensions == (
        "X",
        "Y",
        "Z",
        "Red",
        "Green",
        "Blue",
        "PointSourceId",
        "ContributorCount",
        "Confidence",
    )
    assert capabilities.source_distribution == ((2, 1), (7, 3))


def test_fixture_rejects_missing_contributor_count_dimension(tmp_path):
    path = tmp_path / "standard-only.las"
    points = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    points.x = [0.0]
    points.y = [0.0]
    points.z = [0.0]
    points.write(path)

    with pytest.raises(ValueError, match="canonical ContributorCount"):
        inspect_copc_capabilities(path, 1)
