from __future__ import annotations

import asyncio

import httpx

from package_fixtures import create_package
from viewer.backend.api import create_app


def request(app, method: str, url: str, **kwargs) -> httpx.Response:
    """Use the ASGI transport directly; this environment's TestClient portal hangs."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send())


def test_asset_supports_head_full_and_single_ranges(tmp_path):
    payload = bytes(range(100))
    package = tmp_path / "package"
    create_package(package, payload=payload)
    app = create_app(tmp_path / "catalog.sqlite3")
    app.state.catalog.register_package(package)
    url = "/api/v1/assets/representation_test"

    head = request(app, "HEAD", url)
    assert head.status_code == 200
    assert head.headers["accept-ranges"] == "bytes"
    assert head.headers["content-length"] == "100"
    assert head.content == b""

    response = request(app, "GET", url, headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == payload[10:20]
    assert response.headers["content-range"] == "bytes 10-19/100"
    assert response.headers["content-length"] == "10"
    assert response.headers["cache-control"].endswith("immutable")

    suffix = request(app, "GET", url, headers={"Range": "bytes=-4"})
    assert suffix.status_code == 206
    assert suffix.content == payload[-4:]

    whole = request(app, "GET", url)
    assert whole.status_code == 200
    assert whole.content == payload


def test_asset_rejects_invalid_range_and_detects_post_registration_change(tmp_path):
    package = tmp_path / "package"
    create_package(package, payload=b"0123456789")
    app = create_app(tmp_path / "catalog.sqlite3")
    app.state.catalog.register_package(package)
    url = "/api/v1/assets/representation_test"

    invalid = request(app, "GET", url, headers={"Range": "bytes=99-100"})
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == "bytes */10"

    (package / "geometry" / "points.copc.laz").write_bytes(b"changed-size")
    changed = request(app, "GET", url)
    assert changed.status_code == 409


def test_catalog_bbox_and_detail_endpoints(tmp_path):
    package = tmp_path / "package"
    create_package(package)
    app = create_app(tmp_path / "catalog.sqlite3")
    app.state.catalog.register_package(package)

    listing = request(
        app,
        "GET",
        "/api/v1/catalog/artifacts",
        params={"bbox": "-121.1,36.9,-120.9,37.1"},
    )
    assert listing.status_code == 200
    assert listing.json()["count"] == 1

    detail = request(app, "GET", "/api/v1/catalog/artifacts/artifact_test")
    assert detail.status_code == 200
    representation = detail.json()["representations"][0]
    assert representation["asset_url"].endswith("/representation_test")

    manifest = request(app, "GET", "/api/v1/catalog/artifacts/artifact_test/manifest")
    assert manifest.status_code == 200
    assert manifest.json()["schema_version"] == "1.0.0"
