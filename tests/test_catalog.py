from __future__ import annotations

import sqlite3

import pytest

from package_fixtures import create_package
from viewer.backend.services.catalog import Catalog, CatalogConflict
from viewer.backend.services.package_validator import PackageValidationError


def test_registration_is_idempotent_and_spatial_query_excludes_unaligned(tmp_path):
    aligned_root = tmp_path / "aligned"
    unaligned_root = tmp_path / "unaligned"
    create_package(aligned_root)
    create_package(
        unaligned_root,
        run_id="run_unaligned",
        capture_id="capture_unaligned",
        artifact_id="artifact_unaligned",
        representation_id="representation_unaligned",
        aligned=False,
    )
    catalog = Catalog(tmp_path / "catalog.sqlite3")

    catalog.register_package(aligned_root)
    catalog.register_package(aligned_root)
    catalog.register_package(unaligned_root)

    all_artifacts = catalog.query_artifacts()
    assert {artifact["artifact_id"] for artifact in all_artifacts} == {
        "artifact_test",
        "artifact_unaligned",
    }
    nearby = catalog.query_artifacts(bbox=(-121.1, 36.9, -120.9, 37.1))
    assert [artifact["artifact_id"] for artifact in nearby] == ["artifact_test"]
    elsewhere = catalog.query_artifacts(bbox=(0.0, 0.0, 1.0, 1.0))
    assert elsewhere == []

    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 2
        assert (
            connection.execute("SELECT count(*) FROM representations").fetchone()[0]
            == 2
        )


def test_stable_identity_cannot_be_repointed_to_another_package(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_package(first)
    create_package(second, representation_id="representation_other")
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.register_package(first)

    with pytest.raises(CatalogConflict):
        catalog.register_package(second)


def test_invalid_package_is_not_partially_registered(tmp_path):
    package = tmp_path / "invalid"
    create_package(package)
    (package / "geometry" / "points.copc.laz").write_bytes(b"changed")
    database = tmp_path / "catalog.sqlite3"
    catalog = Catalog(database)

    with pytest.raises(PackageValidationError):
        catalog.register_package(package)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM captures").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
