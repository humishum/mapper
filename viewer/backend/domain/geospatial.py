"""Float64 helpers for deriving catalog placement from artifact-local bounds."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from pyproj import Transformer

from .package import WGS84Footprint


@dataclass(frozen=True)
class GlobalPlacement:
    """Derived WGS84 control-plane placement, always longitude-first."""

    origin_wgs84: tuple[float, float, float]
    footprint_wgs84: WGS84Footprint


def _transform_point(
    matrix: list[float] | tuple[float, ...],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(matrix) != 16 or not all(math.isfinite(value) for value in matrix):
        raise ValueError("transform must be a finite row-major 4x4 matrix")
    x, y, z = point
    transformed = tuple(
        matrix[row * 4] * x
        + matrix[row * 4 + 1] * y
        + matrix[row * 4 + 2] * z
        + matrix[row * 4 + 3]
        for row in range(3)
    )
    if not all(math.isfinite(value) for value in transformed):
        raise ValueError("transformed ECEF coordinate is not finite")
    return transformed


def global_placement_from_local_bounds(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    artifact_local_to_ecef: list[float] | tuple[float, ...],
) -> GlobalPlacement:
    """Derive a conservative WGS84 footprint from all eight local AABB corners.

    The returned origin order is ``(longitude, latitude, ellipsoidal_height_m)``.
    Reconstructions spanning the antimeridian require an explicit split footprint and
    are rejected here rather than producing a misleading nearly-global rectangle.
    """
    if any(not math.isfinite(value) for value in (*bounds_min, *bounds_max)) or any(
        low > high for low, high in zip(bounds_min, bounds_max, strict=True)
    ):
        raise ValueError("local bounds must be finite and ordered")
    ecef_to_wgs84 = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    origin_ecef = _transform_point(artifact_local_to_ecef, (0.0, 0.0, 0.0))
    origin = tuple(float(value) for value in ecef_to_wgs84.transform(*origin_ecef))
    corners = []
    for corner in itertools.product(
        (bounds_min[0], bounds_max[0]),
        (bounds_min[1], bounds_max[1]),
        (bounds_min[2], bounds_max[2]),
    ):
        ecef = _transform_point(artifact_local_to_ecef, corner)
        longitude, latitude, _ = ecef_to_wgs84.transform(*ecef)
        corners.append((float(longitude), float(latitude)))
    longitudes, latitudes = zip(*corners, strict=True)
    if max(longitudes) - min(longitudes) > 180:
        raise ValueError(
            "derived footprint crosses the antimeridian; provide split footprints"
        )
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)
    footprint = WGS84Footprint(
        coordinates=[
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
    )
    return GlobalPlacement(
        origin_wgs84=(origin[0], origin[1], origin[2]),
        footprint_wgs84=footprint,
    )
