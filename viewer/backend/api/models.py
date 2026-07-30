"""Public, renderer-neutral response contracts for the v1 API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.package import (
    AlignmentStatus,
    LayerDefault,
    Manifest,
    SourceRecord,
)


class ApiModel(BaseModel):
    """Response models are closed so accidental database fields never become API."""

    model_config = ConfigDict(extra="ignore")


class GeoJSONPolygon(ApiModel):
    type: Literal["Polygon"]
    coordinates: list[list[tuple[float, float]]]


class ArtifactSummary(ApiModel):
    artifact_id: str
    run_id: str
    capture_id: str
    kind: str
    alignment_status: AlignmentStatus
    frame_name: str
    units: str
    point_count: int | None = None
    footprint: GeoJSONPolygon | None = None
    created_at: datetime


class CatalogArtifactsResponse(ApiModel):
    count: int = Field(ge=0)
    artifacts: list[ArtifactSummary]


class RepresentationResponse(ApiModel):
    representation_id: str
    kind: str
    format: str
    relative_path: str
    media_type: str
    byte_size: int = Field(ge=0)
    sha256: str
    asset_url: str


class ArtifactDetail(ArtifactSummary):
    manifest_sha256: str
    layer_default: LayerDefault | None = None
    representations: list[RepresentationResponse]


class CaptureResponse(ApiModel):
    capture_id: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    device: str | None = None
    lens: str | None = None
    video_name: str | None = None
    source_uri: str | None = None
    frame_count: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    footprint: GeoJSONPolygon | None = None


class RunResponse(ApiModel):
    run_id: str
    capture_id: str
    status: AlignmentStatus
    created_at: datetime
    model_name: str
    model_version: str | None = None
    git_commit: str | None = None
    git_status: Literal["clean", "dirty", "unknown"]
    manifest_sha256: str


class SourcesResponse(ApiModel):
    count: int = Field(ge=0)
    sources: list[SourceRecord]


class HealthResponse(ApiModel):
    status: Literal["healthy"]
    catalog: str


class ErrorResponse(ApiModel):
    detail: str | list[dict[str, Any]]


# Make the manifest contract discoverable under a stable name in OpenAPI.
ManifestResponse = Manifest
