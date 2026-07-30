"""FastAPI routes for the package catalog and immutable data plane."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from ..adapters.blob_store import BlobNotFound, FileBlobStore, UnsafeBlobPath
from ..domain.package import AlignmentStatus, Manifest
from ..services.catalog import Catalog, CatalogNotFound
from .models import (
    ArtifactDetail,
    CaptureResponse,
    CatalogArtifactsResponse,
    ErrorResponse,
    RunResponse,
    SourcesResponse,
)
from .range_response import InvalidRange, iter_file, parse_range_header

router = APIRouter(prefix="/api/v1")


def _catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def _blobs(request: Request) -> FileBlobStore:
    return request.app.state.blobs


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        numbers = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="bbox must contain four comma-separated numbers"
        ) from exc
    if len(numbers) != 4:
        raise HTTPException(
            status_code=422,
            detail="bbox must be min_lon,min_lat,max_lon,max_lat",
        )
    min_lon, min_lat, max_lon, max_lat = numbers
    if not all(math.isfinite(number) for number in numbers):
        raise HTTPException(status_code=422, detail="bbox values must be finite")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise HTTPException(
            status_code=422, detail="bbox longitudes must be between -180 and 180"
        )
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise HTTPException(
            status_code=422, detail="bbox latitudes must be between -90 and 90"
        )
    if min_lat > max_lat:
        raise HTTPException(
            status_code=422,
            detail="bbox minimum latitude must not exceed maximum latitude",
        )
    return numbers


@router.get(
    "/catalog/artifacts",
    response_model=CatalogArtifactsResponse,
    responses={422: {"model": ErrorResponse}},
)
async def query_artifacts(
    request: Request,
    bbox: str | None = None,
    started_after: datetime | None = None,
    ended_before: datetime | None = None,
    alignment_status: AlignmentStatus | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        artifacts = _catalog(request).query_artifacts(
            bbox=_parse_bbox(bbox),
            started_after=started_after,
            ended_before=ended_before,
            alignment_status=(
                alignment_status.value if alignment_status is not None else None
            ),
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"count": len(artifacts), "artifacts": artifacts}


@router.get(
    "/catalog/captures/{capture_id}",
    response_model=CaptureResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_capture(capture_id: str, request: Request):
    try:
        return _catalog(request).get_capture(capture_id)
    except CatalogNotFound as exc:
        raise _not_found(exc) from exc


@router.get(
    "/catalog/runs/{run_id}",
    response_model=RunResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_run(run_id: str, request: Request):
    try:
        return _catalog(request).get_run(run_id)
    except CatalogNotFound as exc:
        raise _not_found(exc) from exc


@router.get(
    "/catalog/artifacts/{artifact_id}",
    response_model=ArtifactDetail,
    responses={404: {"model": ErrorResponse}},
)
async def get_artifact(artifact_id: str, request: Request):
    try:
        artifact = _catalog(request).get_artifact(artifact_id)
    except CatalogNotFound as exc:
        raise _not_found(exc) from exc
    for representation in artifact["representations"]:
        representation["asset_url"] = (
            f"/api/v1/assets/{representation['representation_id']}"
        )
    return artifact


def _etag_matches(value: str | None, etag: str) -> bool:
    if value is None:
        return False
    return any(
        candidate.strip() == "*"
        or candidate.strip().removeprefix("W/") == etag
        for candidate in value.split(",")
    )


def _file_response(
    request: Request,
    *,
    path: Path,
    byte_size: int,
    media_type: str,
    sha256: str,
    head_only: bool,
) -> Response:
    etag = f'"{sha256}"'
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": etag,
    }
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=common_headers)

    range_header = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if range_header and if_range is not None and if_range.strip() != etag:
        range_header = None
    if range_header:
        try:
            byte_range = parse_range_header(range_header, byte_size)
        except InvalidRange:
            return Response(
                status_code=416,
                headers={
                    **common_headers,
                    "Content-Range": f"bytes */{byte_size}",
                    "Content-Length": "0",
                },
            )
        headers = {
            **common_headers,
            "Content-Range": byte_range.content_range,
            "Content-Length": str(byte_range.length),
        }
        if head_only:
            return Response(status_code=206, media_type=media_type, headers=headers)
        return StreamingResponse(
            iter_file(path, start=byte_range.start, length=byte_range.length),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )

    headers = {**common_headers, "Content-Length": str(byte_size)}
    if head_only:
        return Response(status_code=200, media_type=media_type, headers=headers)
    return StreamingResponse(
        iter_file(path, start=0, length=byte_size),
        media_type=media_type,
        headers=headers,
    )


def _manifest_file(artifact_id: str, request: Request) -> tuple[Path, int, str]:
    try:
        artifact = _catalog(request).get_artifact(artifact_id)
    except CatalogNotFound as exc:
        raise _not_found(exc) from exc
    package_root = Path(artifact["package_root"]).resolve()
    path = Path(artifact["manifest_path"]).resolve()
    try:
        path.relative_to(package_root)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="unsafe manifest path") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="manifest file is missing")
    byte_size = path.stat().st_size
    return path, byte_size, artifact["manifest_sha256"]


@router.get(
    "/catalog/artifacts/{artifact_id}/manifest",
    response_model=Manifest,
    responses={
        206: {"description": "Requested manifest byte range"},
        304: {"description": "Manifest ETag matched"},
        404: {"model": ErrorResponse},
        416: {"description": "Requested range is not satisfiable"},
    },
)
async def get_manifest(artifact_id: str, request: Request):
    path, byte_size, checksum = _manifest_file(artifact_id, request)
    return _file_response(
        request,
        path=path,
        byte_size=byte_size,
        media_type="application/json",
        sha256=checksum,
        head_only=False,
    )


@router.head(
    "/catalog/artifacts/{artifact_id}/manifest",
    responses={
        206: {"description": "Requested manifest byte range"},
        304: {"description": "Manifest ETag matched"},
        404: {"model": ErrorResponse},
        416: {"description": "Requested range is not satisfiable"},
    },
)
async def head_manifest(artifact_id: str, request: Request):
    path, byte_size, checksum = _manifest_file(artifact_id, request)
    return _file_response(
        request,
        path=path,
        byte_size=byte_size,
        media_type="application/json",
        sha256=checksum,
        head_only=True,
    )


@router.get(
    "/catalog/artifacts/{artifact_id}/sources",
    response_model=SourcesResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_sources(artifact_id: str, request: Request):
    try:
        sources = _catalog(request).get_sources(artifact_id)
    except CatalogNotFound as exc:
        raise _not_found(exc) from exc
    return {"count": len(sources), "sources": sources}


def _asset_response(
    representation_id: str, request: Request, *, head_only: bool
) -> Response:
    try:
        blob = _blobs(request).resolve(representation_id)
    except BlobNotFound as exc:
        raise _not_found(exc) from exc
    except UnsafeBlobPath as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _file_response(
        request,
        path=blob.path,
        byte_size=blob.byte_size,
        media_type=blob.media_type,
        sha256=blob.sha256,
        head_only=head_only,
    )


@router.get(
    "/assets/{representation_id}",
    responses={
        206: {"description": "Requested representation byte range"},
        304: {"description": "Representation ETag matched"},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        416: {"description": "Requested range is not satisfiable"},
    },
)
async def get_asset(representation_id: str, request: Request):
    return _asset_response(representation_id, request, head_only=False)


@router.head("/assets/{representation_id}")
async def head_asset(representation_id: str, request: Request):
    return _asset_response(representation_id, request, head_only=True)
