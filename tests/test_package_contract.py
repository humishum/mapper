from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer
from pydantic import ValidationError

from package_fixtures import IDENTITY, create_package
from viewer.backend.domain.geospatial import global_placement_from_local_bounds
from viewer.backend.domain.package import (
    Alignment,
    ArtifactFile,
    CaptureMetadata,
    CoordinateFrame,
    Manifest,
    Producer,
    TABULAR_COLUMN_DTYPES,
    WGS84Footprint,
)
from viewer.backend.services.package_validator import (
    PackageValidationError,
    PackageValidator,
)
from viewer.backend.services.package_writer import describe_artifact, write_manifest


def test_unaligned_manifest_cannot_claim_global_placement(tmp_path):
    package = tmp_path / "bad"
    package.mkdir()
    valid = create_package(tmp_path / "valid", aligned=False)
    data = valid.model_dump(mode="json", by_alias=True)
    data["coordinate_frame"]["origin_wgs84"] = [-121.0, 37.0, 100.0]

    with pytest.raises(ValidationError, match="unaligned artifacts cannot claim"):
        Manifest.model_validate(data)


def test_aligned_manifest_requires_footprint():
    with pytest.raises(ValidationError, match="require footprint_wgs84"):
        Manifest(
            run_id="run",
            capture_id="capture",
            artifact_id="artifact",
            created_at="2026-07-25T00:00:00Z",
            capture=CaptureMetadata(),
            producer=Producer(
                model_name="model",
                adapter_name="adapter",
                adapter_version="1",
                publisher_name="publisher",
                publisher_version="1",
            ),
            coordinate_frame=CoordinateFrame(
                units="metre",
                axis_order=["east", "north", "up"],
                handedness="right",
                origin_wgs84=(-121, 37, 0),
                transform_to_ecef=IDENTITY,
            ),
            alignment=Alignment(
                status="aligned",
                method="test",
                model_to_artifact_local=IDENTITY,
                scale=1,
                inlier_count=4,
            ),
            artifacts=[
                ArtifactFile(
                    representation_id="representation",
                    kind="points",
                    format="copc/laz",
                    path="geometry/points.copc.laz",
                    media_type="application/vnd.laszip",
                    byte_size=0,
                    sha256="0" * 64,
                    frame="artifact_local",
                    bounds_min=(0, 0, 0),
                    bounds_max=(1, 1, 1),
                    point_count=1,
                    required_dimensions=["X", "Y", "Z"],
                )
            ],
        )


def test_validator_accepts_valid_package_and_detects_tampering(tmp_path):
    package = tmp_path / "package"
    create_package(package)
    validator = PackageValidator()

    result = validator.validate(package)
    assert result.manifest.artifact_id == "artifact_test"
    assert len(result.manifest_sha256) == 64

    (package / "geometry" / "points.copc.laz").write_bytes(b"tampered")
    with pytest.raises(PackageValidationError) as error:
        validator.validate(package)
    assert {issue.code for issue in error.value.issues} == {
        "byte_size_mismatch",
        "checksum_mismatch",
    }


def test_validator_rejects_symlink_escape(tmp_path):
    package = tmp_path / "package"
    create_package(package)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = package / "geometry" / "escape.bin"
    link.symlink_to(outside)
    manifest_path = package / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["artifacts"][0].update(
        {
            "path": "geometry/escape.bin",
            "byte_size": outside.stat().st_size,
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PackageValidationError) as error:
        PackageValidator().validate(package)
    assert "path_escape" in {issue.code for issue in error.value.issues}


def test_generated_schema_is_current():
    schema = json.loads(open("schemas/manifest.v1.json", encoding="utf-8").read())
    assert schema["$id"].endswith("/manifest.v1.json")
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"


def test_wgs84_footprint_is_closed():
    with pytest.raises(ValidationError, match="must be closed"):
        WGS84Footprint(
            coordinates=[
                (-121.0, 37.0),
                (-120.9, 37.0),
                (-120.9, 37.1),
                (-121.0, 37.1),
            ]
        )


def test_global_placement_helper_is_longitude_first():
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    ecef_x, ecef_y, ecef_z = to_ecef.transform(-121.0, 37.0, 100.0)
    local_to_ecef = [
        1.0,
        0.0,
        0.0,
        ecef_x,
        0.0,
        1.0,
        0.0,
        ecef_y,
        0.0,
        0.0,
        1.0,
        ecef_z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    placement = global_placement_from_local_bounds(
        (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), local_to_ecef
    )

    longitude, latitude, height = placement.origin_wgs84
    assert longitude == pytest.approx(-121.0)
    assert latitude == pytest.approx(37.0)
    assert height == pytest.approx(100.0)
    assert placement.footprint_wgs84.bounds[0] < -121.0
    assert placement.footprint_wgs84.bounds[2] > -121.0


def _add_poses_parquet(
    package,
    manifest,
    *,
    timestamp_type=pa.float64(),
    omit_column=None,
):
    physical_columns = {}
    for name, dtype in TABULAR_COLUMN_DTYPES["poses"].items():
        if name == omit_column:
            continue
        arrow_type = (
            timestamp_type
            if name == "timestamp_s"
            else (pa.int64() if dtype == "int64" else pa.float64())
        )
        value = 0 if pa.types.is_integer(arrow_type) else 0.0
        physical_columns[name] = pa.array([value], type=arrow_type)
    path = package / "cameras" / "poses.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(physical_columns), path)
    artifact = describe_artifact(
        package,
        "cameras/poses.parquet",
        representation_id="representation_poses",
        kind="poses",
        format="parquet",
        media_type="application/vnd.apache.parquet",
        frame="artifact_local",
        columns=[
            {"name": name, "dtype": dtype}
            for name, dtype in TABULAR_COLUMN_DTYPES["poses"].items()
        ],
    )
    manifest.artifacts.append(artifact)
    write_manifest(package, manifest)


def test_validator_inspects_physical_parquet_schema(tmp_path):
    package = tmp_path / "package"
    manifest = create_package(package)
    _add_poses_parquet(package, manifest)

    result = PackageValidator().validate(package)

    assert {artifact.kind for artifact in result.manifest.artifacts} == {
        "points",
        "poses",
    }


@pytest.mark.parametrize(
    ("timestamp_type", "omit_column", "expected_code"),
    [
        (pa.int64(), None, "parquet_dtype_mismatch"),
        (pa.float64(), "cx", "parquet_column_missing"),
    ],
)
def test_validator_rejects_physical_parquet_contract_mismatch(
    tmp_path, timestamp_type, omit_column, expected_code
):
    package = tmp_path / "package"
    manifest = create_package(package)
    _add_poses_parquet(
        package,
        manifest,
        timestamp_type=timestamp_type,
        omit_column=omit_column,
    )

    with pytest.raises(PackageValidationError) as error:
        PackageValidator().validate(package)

    assert expected_code in {issue.code for issue in error.value.issues}
