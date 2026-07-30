from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from package_fixtures import create_package
from viewer.backend.api import create_app
from viewer.backend.api.export_openapi import export_openapi
from viewer.backend.domain.package import WGS84Footprint
from viewer.backend.services.package_writer import write_manifest


def request(app, method: str, url: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send())


def create_site(root, *, suffix: str, longitude: float, aligned: bool = True):
    manifest = create_package(
        root,
        run_id=f"run_{suffix}",
        capture_id=f"capture_{suffix}",
        artifact_id=f"artifact_{suffix}",
        representation_id=f"representation_{suffix}",
        aligned=aligned,
    )
    if aligned:
        manifest.coordinate_frame.origin_wgs84 = (longitude, 5.0, 10.0)
        manifest.footprint_wgs84 = WGS84Footprint(
            coordinates=[
                (longitude - 0.1, 4.9),
                (longitude + 0.1, 4.9),
                (longitude + 0.1, 5.1),
                (longitude - 0.1, 5.1),
                (longitude - 0.1, 4.9),
            ]
        )
        write_manifest(root, manifest)
    return manifest


def test_crossing_antimeridian_bbox_and_unaligned_discovery(tmp_path):
    catalog_path = tmp_path / "catalog.sqlite3"
    app = create_app(catalog_path)
    for suffix, longitude, aligned in (
        ("east", 179.5, True),
        ("west", -179.5, True),
        ("local", 0.0, False),
    ):
        root = tmp_path / suffix
        create_site(root, suffix=suffix, longitude=longitude, aligned=aligned)
        app.state.catalog.register_package(root)

    crossing = request(
        app,
        "GET",
        "/api/v1/catalog/artifacts",
        params={"bbox": "179,-10,-179,10"},
    )
    assert crossing.status_code == 200
    assert {
        item["artifact_id"] for item in crossing.json()["artifacts"]
    } == {"artifact_east", "artifact_west"}

    local = request(
        app,
        "GET",
        "/api/v1/catalog/artifacts",
        params={"alignment_status": "unaligned"},
    )
    assert [item["artifact_id"] for item in local.json()["artifacts"]] == [
        "artifact_local"
    ]
    assert local.json()["artifacts"][0]["footprint"] is None


@pytest.mark.parametrize(
    ("bbox", "detail"),
    [
        ("nan,0,1,1", "bbox values must be finite"),
        ("-181,0,1,1", "bbox longitudes must be between -180 and 180"),
        ("0,-91,1,1", "bbox latitudes must be between -90 and 90"),
        ("0,5,1,4", "bbox minimum latitude must not exceed maximum latitude"),
    ],
)
def test_bbox_errors_are_precise(tmp_path, bbox, detail):
    app = create_app(tmp_path / "catalog.sqlite3")
    response = request(
        app, "GET", "/api/v1/catalog/artifacts", params={"bbox": bbox}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": detail}


def test_manifest_ranges_conditionals_and_public_detail_contract(tmp_path):
    package = tmp_path / "package"
    create_package(package)
    app = create_app(tmp_path / "catalog.sqlite3")
    app.state.catalog.register_package(package)
    url = "/api/v1/catalog/artifacts/artifact_test/manifest"

    full = request(app, "GET", url)
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert full.json()["artifact_id"] == "artifact_test"

    partial = request(app, "GET", url, headers={"Range": "bytes=0-15"})
    assert partial.status_code == 206
    assert partial.content == full.content[:16]
    assert partial.headers["content-range"].startswith("bytes 0-15/")

    uppercase_unit = request(app, "GET", url, headers={"Range": "BYTES=-8"})
    assert uppercase_unit.status_code == 206
    assert uppercase_unit.content == full.content[-8:]

    unchanged = request(
        app,
        "GET",
        url,
        headers={"If-None-Match": f'W/{full.headers["etag"]}'},
    )
    assert unchanged.status_code == 304

    detail = request(app, "GET", "/api/v1/catalog/artifacts/artifact_test")
    assert detail.status_code == 200
    assert "package_root" not in detail.json()
    assert "manifest_path" not in detail.json()
    assert detail.json()["representations"][0]["asset_url"].startswith("/api/v1/")


def test_openapi_has_typed_v1_contracts_and_no_legacy_routes(tmp_path):
    schema = create_app(tmp_path / "catalog.sqlite3").openapi()
    paths = schema["paths"]
    assert "/api/locations" not in paths
    assert "/api/pointcloud/{location_name}" not in paths
    listing_schema = paths["/api/v1/catalog/artifacts"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert listing_schema["$ref"].endswith("/CatalogArtifactsResponse")
    manifest_schema = paths[
        "/api/v1/catalog/artifacts/{artifact_id}/manifest"
    ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert manifest_schema["$ref"].endswith("/Manifest")


def test_checked_in_openapi_export_is_deterministic(tmp_path):
    exported = export_openapi(tmp_path / "openapi.json")
    checked_in = Path("schemas/openapi.v1.json")
    assert exported.read_bytes() == checked_in.read_bytes()


def test_default_catalog_path_is_resolved_when_app_is_created(tmp_path, monkeypatch):
    catalog_path = tmp_path / "from-environment.sqlite3"
    monkeypatch.setenv("MAPPER_CATALOG_PATH", str(catalog_path))

    app = create_app()

    assert app.state.catalog.database_path == catalog_path
    assert catalog_path.is_file()
